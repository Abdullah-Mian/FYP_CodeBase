/*
 * ESP32 Audio Streamer to Raspberry Pi 4 - V4 (PRODUCTION)
 * =========================================================
 * Compatible with: ESP32 Arduino Core 3.3.3 (IDF 5.5)
 *                  Arduino IDE 2.3.7
 *                  Board: ESP32 Dev Module (NodeMCU-32S)
 *
 * Fixed in V4:
 *   - Calibration no longer captures wake-up sound as noise floor
 *   - Uses hardcoded baseline + verification instead of blind calibration
 *   - ADC settling period (1s) after deep sleep wake
 *   - Ring buffer drain before VAD processing begins
 *   - Proper silence detection with hysteresis
 *
 * Hardcoded from 3 diagnostic runs (YOUR hardware):
 *   DC Offset:    ~2086 (average across runs)
 *   Noise RMS:    ~2.5 counts (true silence)
 *   Speech RMS:   23-82 counts (distance dependent)
 *   VAD Threshold: 8.0 RMS (safely above noise, below speech)
 *
 * Wiring:
 *   Mic AO  -> GPIO36 (VP, ADC1_CH0)
 *   Mic D0  -> GPIO33 (RTC wake-up)
 *   Mic +   -> 3.3V
 *   Mic GND -> GND
 *   ESP32 GPIO17 (TX2) -> RPi4 GPIO15 (RXD, Pin 10)
 *   ESP32 GND -> RPi4 GND
 *   ESP32 5V  <- RPi4 5V (Pin 2)
 */

#include <esp_sleep.h>
#include <driver/adc.h>

// ==================== PIN CONFIG =======================
#define MIC_ADC_CHANNEL      ADC1_CHANNEL_0   // GPIO36
#define WAKEUP_PIN           GPIO_NUM_33      // D0 from mic module
#define UART_TX_PIN          17               // TX2 -> RPi4 RXD

// ==================== AUDIO CONFIG =====================
#define SAMPLE_RATE          16000
#define TIMER_INTERVAL_US    63               // ~15873 Hz ≈ 16kHz
#define UART_BAUD            921600

// ==================== BUFFER CONFIG ====================
#define CHUNK_SIZE           512
#define RING_BUF_SIZE        4096             // Must be power of 2

// ==================== HARDCODED CALIBRATION =============
// From your 3 diagnostic runs (measured in true silence):
//   Run 1: DC=2087.5, Noise=3.39
//   Run 2: DC=2086.2, Noise=2.58
//   Run 3: DC=2085.2, Noise=2.40
// Average:
#define FIXED_DC_OFFSET      2086.0f
#define FIXED_NOISE_RMS      3.0f             // Worst case from your runs
#define FIXED_VAD_THRESHOLD  8.0f             // ~3x noise, well below speech RMS of ~23+

// ==================== VAD CONFIG =======================
#define SILENCE_TIMEOUT_MS   3000             // Sleep after 3s continuous silence
#define VOICE_CONFIRM_CHUNKS 3                // Need 3 consecutive voice chunks (~96ms)
#define SETTLE_TIME_MS       1000             // ADC settling after deep sleep
#define DRAIN_TIME_MS        200              // Drain ring buffer before processing

// ==================== PROTOCOL =========================
static const uint8_t SYNC_HEADER[2] = {0xAA, 0x55};

// ==================== RING BUFFER ======================
static volatile int16_t ring_buf[RING_BUF_SIZE];
static volatile uint32_t wr_pos = 0;
static volatile uint32_t rd_pos = 0;

// ==================== TIMER ============================
hw_timer_t *tmr = NULL;

void IRAM_ATTR onTimer() {
  int raw = adc1_get_raw(MIC_ADC_CHANNEL);
  uint32_t nxt = (wr_pos + 1) & (RING_BUF_SIZE - 1);
  if (nxt != rd_pos) {
    ring_buf[wr_pos] = (int16_t)raw;
    wr_pos = nxt;
  }
  // No overflow flag - we just drop samples silently during settling
}

// ==================== STATE ============================
static float dc_offset = FIXED_DC_OFFSET;
static float vad_threshold = FIXED_VAD_THRESHOLD;
static unsigned long last_voice_ms = 0;
static unsigned long boot_time_ms = 0;
static bool is_streaming = false;
static int voice_confirm_count = 0;
static bool settled = false;

// ==================== BUFFERS ==========================
static int16_t raw_chunk[CHUNK_SIZE];
static int16_t pcm_chunk[CHUNK_SIZE];

// ==================== HELPERS ==========================

inline uint32_t samples_available() {
  uint32_t w = wr_pos;
  uint32_t r = rd_pos;
  return (w >= r) ? (w - r) : (RING_BUF_SIZE - r + w);
}

void read_chunk_from_ring(int16_t *dst, uint32_t n) {
  for (uint32_t i = 0; i < n; i++) {
    dst[i] = ring_buf[rd_pos];
    rd_pos = (rd_pos + 1) & (RING_BUF_SIZE - 1);
  }
}

void drain_ring_buffer() {
  // Discard all samples currently in the buffer
  rd_pos = wr_pos;
}

// ==================== VERIFY DC OFFSET =================
void verify_dc_offset() {
  /*
   * After settling, take a quick measurement to fine-tune DC offset.
   * But CLAMP the result: if it's wildly different from expected,
   * the wake-up sound is still happening — use the hardcoded value.
   */
  uint32_t n = samples_available();
  if (n < 256) return; // Not enough samples
  if (n > 1024) n = 1024;

  int16_t *buf = (int16_t *)malloc(n * sizeof(int16_t));
  if (!buf) return;

  read_chunk_from_ring(buf, n);

  // Compute DC
  int64_t sum = 0;
  for (uint32_t i = 0; i < n; i++) sum += buf[i];
  float measured_dc = (float)sum / n;

  // Compute RMS
  float sq_sum = 0;
  for (uint32_t i = 0; i < n; i++) {
    float v = (float)buf[i] - measured_dc;
    sq_sum += v * v;
  }
  float measured_rms = sqrtf(sq_sum / n);

  free(buf);

  // Only accept if DC is within ±20 of expected AND rms is reasonable
  // (if rms > 15, the wake-up sound is still echoing — reject)
  if (fabsf(measured_dc - FIXED_DC_OFFSET) < 20.0f && measured_rms < 15.0f) {
    dc_offset = measured_dc;
    Serial.printf("[CAL] DC verified: %.1f (RMS=%.2f, using measured)\n",
                  dc_offset, measured_rms);
  } else {
    dc_offset = FIXED_DC_OFFSET;
    Serial.printf("[CAL] DC check: measured=%.1f RMS=%.2f (noisy, using hardcoded %.1f)\n",
                  measured_dc, measured_rms, FIXED_DC_OFFSET);
  }
}

// ==================== VAD ==============================
float compute_rms(int16_t *samples, uint32_t n) {
  float sq_sum = 0;
  for (uint32_t i = 0; i < n; i++) {
    float v = (float)samples[i] - dc_offset;
    sq_sum += v * v;
  }
  return sqrtf(sq_sum / n);
}

// ==================== PCM CONVERSION ===================
void convert_pcm(int16_t *raw, int16_t *pcm, uint32_t n) {
  for (uint32_t i = 0; i < n; i++) {
    int32_t centered = (int32_t)raw[i] - (int32_t)dc_offset;
    int32_t scaled = centered << 8; // ×256: 12-bit -> fills 16-bit range

    if (scaled > 32767) scaled = 32767;
    if (scaled < -32768) scaled = -32768;

    pcm[i] = (int16_t)scaled;
  }
}

// ==================== UART SEND ========================
void send_frame(int16_t *pcm, uint32_t sample_count) {
  uint16_t byte_len = sample_count * 2;
  uint8_t hdr[4] = {
    SYNC_HEADER[0], SYNC_HEADER[1],
    (uint8_t)(byte_len >> 8), (uint8_t)(byte_len & 0xFF)
  };
  Serial2.write(hdr, 4);
  Serial2.write((const uint8_t *)pcm, byte_len);
}

void send_eos() {
  uint8_t eos[4] = {0xAA, 0x55, 0x00, 0x00};
  Serial2.write(eos, 4);
  Serial2.flush();
}

// ==================== DEEP SLEEP =======================
void go_to_sleep() {
  Serial.println("[ESP32] Going to deep sleep...");
  Serial.flush();

  if (tmr) {
    timerDetachInterrupt(tmr);
    timerEnd(tmr);
    tmr = NULL;
  }

  esp_sleep_enable_ext0_wakeup(WAKEUP_PIN, 1);
  delay(50);
  esp_deep_sleep_start();
}

// ==================== SETUP ============================
void setup() {
  Serial.begin(115200);
  Serial2.begin(UART_BAUD, SERIAL_8N1, -1, UART_TX_PIN);

  boot_time_ms = millis();

  // Wake-up reason
  esp_sleep_wakeup_cause_t reason = esp_sleep_get_wakeup_cause();
  if (reason == ESP_SLEEP_WAKEUP_EXT0) {
    Serial.println("[ESP32] WOKE UP from sound!");
  } else {
    Serial.println("[ESP32] Fresh boot / reset");
  }

  // Configure ADC
  adc1_config_width(ADC_WIDTH_BIT_12);
  adc1_config_channel_atten(MIC_ADC_CHANNEL, ADC_ATTEN_DB_11);
  pinMode(WAKEUP_PIN, INPUT);

  // Reset ring buffer
  wr_pos = 0;
  rd_pos = 0;

  // Start timer (samples will accumulate but we'll discard them during settling)
  tmr = timerBegin(1000000);
  timerAttachInterrupt(tmr, &onTimer);
  timerAlarm(tmr, TIMER_INTERVAL_US, true, 0);

  // Use hardcoded calibration values
  dc_offset = FIXED_DC_OFFSET;
  vad_threshold = FIXED_VAD_THRESHOLD;
  settled = false;
  is_streaming = false;
  voice_confirm_count = 0;

  Serial.printf("[ESP32] Settling for %dms (discarding wake-up noise)...\n", SETTLE_TIME_MS);
}

// ==================== LOOP =============================
void loop() {
  unsigned long now = millis();
  unsigned long elapsed = now - boot_time_ms;

  // ============ PHASE 1: SETTLING (first 1 second) ============
  // Discard all ADC data — the wake-up sound and ADC settling noise
  if (!settled) {
    if (elapsed < SETTLE_TIME_MS) {
      // Keep draining the buffer — throw away everything
      drain_ring_buffer();
      delay(10);
      return;
    }

    // Settling done — drain one more time, then verify DC
    drain_ring_buffer();
    delay(DRAIN_TIME_MS); // Let fresh quiet samples accumulate

    verify_dc_offset();

    settled = true;
    last_voice_ms = now;

    Serial.printf("[ESP32] Ready | DC=%.1f | Thr=%.1f | Listening...\n",
                  dc_offset, vad_threshold);
    return;
  }

  // ============ PHASE 2: NORMAL OPERATION ============
  if (samples_available() < CHUNK_SIZE) {
    delay(1);
    return;
  }

  // Read chunk
  read_chunk_from_ring(raw_chunk, CHUNK_SIZE);

  // Compute RMS for this chunk
  float rms = compute_rms(raw_chunk, CHUNK_SIZE);
  bool voice = (rms > vad_threshold);

  if (voice) {
    voice_confirm_count++;
    last_voice_ms = millis();

    if (!is_streaming && voice_confirm_count >= VOICE_CONFIRM_CHUNKS) {
      is_streaming = true;
      Serial.printf("[ESP32] >>> STREAMING (RMS=%.1f)\n", rms);
    }
  } else {
    // Reset confirmation counter only if RMS is well below threshold
    // This provides hysteresis — short dips don't reset the counter
    if (rms < vad_threshold * 0.6f) {
      voice_confirm_count = 0;
    }
  }

  // Send audio while streaming
  if (is_streaming) {
    convert_pcm(raw_chunk, pcm_chunk, CHUNK_SIZE);
    send_frame(pcm_chunk, CHUNK_SIZE);

    // Check silence timeout
    unsigned long silence_duration = millis() - last_voice_ms;
    if (silence_duration > SILENCE_TIMEOUT_MS) {
      Serial.printf("[ESP32] >>> SILENCE (%lums) - Ending session\n", silence_duration);
      send_eos();
      is_streaming = false;
      voice_confirm_count = 0;
      delay(100);
      go_to_sleep();
    }
  }

  // False wake-up protection
  // If 10 seconds pass and we never started streaming, go back to sleep
  if (!is_streaming && (millis() - boot_time_ms) > 10000) {
    Serial.println("[ESP32] No voice in 10s since boot - sleeping");
    go_to_sleep();
  }
}