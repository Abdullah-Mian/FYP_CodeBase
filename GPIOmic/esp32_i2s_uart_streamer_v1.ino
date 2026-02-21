/*
 * ESP32 I2S MEMS Mic -> UART Streamer (DMA-based) + Deep Sleep Wake on Clap
 * =========================================================================
 * Arduino-ESP32 core: 3.3.3
 *
 * Audio source: I2S MEMS mic (INMP441 / ICS-43434 / SPH0645)
 * Wake source:  KY-037 (or LM393 sound trigger) D0 -> RTC GPIO (ext0)
 *
 * Protocol to Pi:
 *   [0xAA][0x55][LEN_H][LEN_L][PCM_BYTES...]
 *   LEN=0 means end-of-stream
 *
 * Notes:
 * - Audio is 16 kHz mono 16-bit signed PCM
 * - I2S uses DMA internally (ESP-IDF driver)
 */

#include <Arduino.h>
#include "driver/i2s.h"
#include "esp_sleep.h"

// ================= UART to Pi =================
#define UART_BAUD     921600
#define UART_TX_PIN   17   // ESP32 TX2 -> Pi RXD

// ================= Wake pin (from KY-037 D0) =================
#define WAKEUP_PIN    GPIO_NUM_32  // RTC-capable pin. Connect KY-037 D0 here.

// ================= I2S Mic pins =================
// Suggested wiring:
#define I2S_BCLK      26
#define I2S_LRCLK     25
#define I2S_DIN       33  // Mic SD (DOUT) -> ESP32 DIN

// ================= Audio format =================
#define SAMPLE_RATE   16000
#define I2S_PORT      I2S_NUM_0

// DMA settings: tune for latency vs robustness
#define DMA_BUF_COUNT 6
#define DMA_BUF_LEN   256   // samples per DMA buffer (per channel); small => low latency

// Chunk we transmit per frame (in samples)
#define CHUNK_SAMPLES 512   // 32ms at 16kHz

// VAD / session control
#define VAD_START_RMS       1200   // start streaming if RMS above this
#define VAD_STOP_RMS        800    // consider silence if RMS below this (hysteresis)
#define SILENCE_TIMEOUT_MS  2000   // sleep after 2s silence
#define START_CONFIRM_CHUNKS 3     // require N chunks above threshold to start

static const uint8_t SYNC_HEADER[2] = {0xAA, 0x55};

// Buffers
static int16_t i2s_chunk[CHUNK_SAMPLES];   // raw I2S samples (16-bit)
static bool streaming = false;
static int start_confirm = 0;
static unsigned long last_voice_ms = 0;

// ---------- Helpers ----------
static inline float compute_rms_i16(const int16_t* x, size_t n) {
  // RMS = sqrt(mean(x^2))
  // Use 64-bit accumulator to avoid overflow
  int64_t sum_sq = 0;
  for (size_t i = 0; i < n; i++) {
    int32_t v = x[i];
    sum_sq += (int64_t)v * v;
  }
  float mean = (float)sum_sq / (float)n;
  return sqrtf(mean);
}

static void send_frame_pcm(const int16_t* samples, size_t sample_count) {
  uint16_t byte_len = (uint16_t)(sample_count * sizeof(int16_t));
  uint8_t hdr[4] = { SYNC_HEADER[0], SYNC_HEADER[1], (uint8_t)(byte_len >> 8), (uint8_t)(byte_len & 0xFF) };
  Serial2.write(hdr, 4);
  Serial2.write((const uint8_t*)samples, byte_len);
}

static void send_eos() {
  uint8_t eos[4] = {0xAA, 0x55, 0x00, 0x00};
  Serial2.write(eos, 4);
  Serial2.flush();
}

static void go_to_sleep() {
  Serial.println("[ESP32] Deep sleeping...");
  Serial.flush();

  // Stop I2S
  i2s_stop(I2S_PORT);
  i2s_driver_uninstall(I2S_PORT);

  // Configure wakeup on HIGH
  esp_sleep_enable_ext0_wakeup(WAKEUP_PIN, 1);

  delay(50);
  esp_deep_sleep_start();
}

// ---------- I2S init ----------
static void setup_i2s_mic() {
  // I2S config for standard I2S mics that output 24-bit left-justified inside 32-bit frames
  // Many INMP441 breakouts work well with 16-bit read too (driver will truncate/shift).
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
  cfg.sample_rate = SAMPLE_RATE;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;    // mono
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = DMA_BUF_COUNT;
  cfg.dma_buf_len = DMA_BUF_LEN;
  cfg.use_apll = false;
  cfg.tx_desc_auto_clear = false;
  cfg.fixed_mclk = 0;

  i2s_pin_config_t pins = {};
  pins.bck_io_num = I2S_BCLK;
  pins.ws_io_num = I2S_LRCLK;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num = I2S_DIN;

  ESP_ERROR_CHECK(i2s_driver_install(I2S_PORT, &cfg, 0, NULL));
  ESP_ERROR_CHECK(i2s_set_pin(I2S_PORT, &pins));
  ESP_ERROR_CHECK(i2s_zero_dma_buffer(I2S_PORT));
  ESP_ERROR_CHECK(i2s_start(I2S_PORT));
}

void setup() {
  Serial.begin(115200);
  Serial2.begin(UART_BAUD, SERIAL_8N1, -1, UART_TX_PIN);

  pinMode((int)WAKEUP_PIN, INPUT);

  auto cause = esp_sleep_get_wakeup_cause();
  if (cause == ESP_SLEEP_WAKEUP_EXT0) {
    Serial.println("[ESP32] Wake from clap trigger (ext0).");
  } else {
    Serial.println("[ESP32] Boot/reset.");
  }

  setup_i2s_mic();

  streaming = false;
  start_confirm = 0;
  last_voice_ms = millis();

  Serial.println("[ESP32] I2S mic started, waiting for voice...");
}

void loop() {
  // Read exactly CHUNK_SAMPLES samples from I2S.
  // bytes_read will be multiple of 2.
  size_t bytes_read = 0;
  esp_err_t ok = i2s_read(I2S_PORT, (void*)i2s_chunk, sizeof(i2s_chunk), &bytes_read, portMAX_DELAY);
  if (ok != ESP_OK || bytes_read == 0) return;

  size_t samples_read = bytes_read / sizeof(int16_t);

  float rms = compute_rms_i16(i2s_chunk, samples_read);

  // VAD with hysteresis + confirmation
  if (!streaming) {
    if (rms > VAD_START_RMS) {
      start_confirm++;
      if (start_confirm >= START_CONFIRM_CHUNKS) {
        streaming = true;
        last_voice_ms = millis();
        Serial.printf("[ESP32] STREAM START (rms=%.0f)\n", rms);
      }
    } else {
      start_confirm = 0;
    }
  } else {
    if (rms > VAD_STOP_RMS) {
      last_voice_ms = millis();
    }
  }

  // Send audio if streaming (also send trailing audio while we wait for silence timeout)
  if (streaming) {
    send_frame_pcm(i2s_chunk, samples_read);

    if ((millis() - last_voice_ms) > SILENCE_TIMEOUT_MS) {
      Serial.println("[ESP32] STREAM STOP (silence).");
      send_eos();
      delay(100);
      go_to_sleep();
    }
  } else {
    // If we woke up falsely (clap triggered but no speech), go back to sleep
    if ((millis() - last_voice_ms) > 8000) {
      Serial.println("[ESP32] No voice after wake. Sleeping.");
      go_to_sleep();
    }
  }
}