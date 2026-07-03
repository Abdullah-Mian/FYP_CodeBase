/*
 * ESP32 WROOM32 — Gemini Voice Assistant  (v3)
 * =============================================
 * Wake:     GPIO35 HIGH (KY-037 DO) → ext0 deep-sleep wakeup
 * Mic:      INMP441  SCK=14, WS=15, SD=32, L/R=GND  (I2S_NUM_1)
 * Speaker:  PAM8403  DAC=GPIO25 via RC filter         (I2S_NUM_0 built-in DAC)
 *
 * Fixes vs v2:
 *   1. INMP441 shift changed from >>11 to >>14 — was constantly clipping at 32767
 *   2. Cold boot → immediate deep sleep, ONLY GPIO35 wakeup starts the pipeline
 *   3. No-response timeout: plays 3 beeps on DAC then sleeps
 *   4. Short-clip EOS from server: plays 3 beeps then sleeps
 *   5. Server reply timeout: 25 s hard cap then beep + sleep
 *
 * Frame protocol (both directions):
 *   Audio : [0xAA][0x55][LEN_H][LEN_L][int16 PCM bytes...]
 *   EOS   : [0xAA][0x55][0x00][0x00]
 */

#include <WiFi.h>
#include <WiFiUDP.h>
#include <driver/i2s.h>
#include <esp_sleep.h>
#include <math.h>

// ─── Network ─────────────────────────────────────────────────────────────────
const char* SSID      = "Victus";
const char* PASSWORD  = "68986898";
const char* SERVER_IP = "192.168.137.1";
const int   MIC_PORT  = 5008;
const int   SPK_PORT  = 5009;
// ─────────────────────────────────────────────────────────────────────────────

// ─── Pins ────────────────────────────────────────────────────────────────────
#define WAKE_PIN    GPIO_NUM_35   // KY-037 DO → ext0 wakeup (input-only, RTC)
#define I2S_MIC     I2S_NUM_1
#define I2S_DAC     I2S_NUM_0
#define MIC_SCK     14
#define MIC_WS      15
#define MIC_SD      32
// ─────────────────────────────────────────────────────────────────────────────

// ─── Audio ───────────────────────────────────────────────────────────────────
#define SAMPLE_RATE     16000
#define CHUNK_SAMPLES   256       // 16 ms per DMA chunk
// ─────────────────────────────────────────────────────────────────────────────

// ─── VAD ─────────────────────────────────────────────────────────────────────
// RMS=32767 constantly → mic was clipping (shift was >>11, now >>14)
// Normal speech after fix: RMS ~1000–8000. Tune VAD_SPEECH_RMS if needed.
#define VAD_SPEECH_RMS      800    // int16 RMS above this = voice
#define SILENCE_TIMEOUT_MS  2000   // ms of silence after speech → send EOS
#define MAX_STREAM_MS       8000   // hard cap: force EOS regardless
// ─────────────────────────────────────────────────────────────────────────────

// ─── Timeouts ────────────────────────────────────────────────────────────────
#define SERVER_REPLY_TIMEOUT_MS  25000  // wait this long for first TTS packet
#define PLAY_DONE_TIMEOUT_MS       800  // ms after last TTS packet → playback done
// ─────────────────────────────────────────────────────────────────────────────

typedef enum { STREAMING, WAIT_REPLY, PLAYING } AppState;
AppState state = STREAMING;

WiFiUDP udpMic;
WiFiUDP udpSpk;

static int32_t  micRaw[CHUNK_SAMPLES];
static int16_t  micPcm[CHUNK_SAMPLES];
static uint8_t  txBuf[4 + CHUNK_SAMPLES * 2];
static uint8_t  rxBuf[4 + CHUNK_SAMPLES * 2];
static uint16_t dacBuf[CHUNK_SAMPLES * 2];

static uint32_t streamStartMs   = 0;
static uint32_t lastVoiceMs     = 0;
static uint32_t lastPktMs       = 0;
static uint32_t waitReplyStart  = 0;
static uint32_t totalPktsSent   = 0;
static bool     speechDetected  = false;

// ─────────────────────────────────────────────────────────────────────────────
// Beep: plays N short tones on DAC so the user hears feedback
// freq_hz: tone frequency  dur_ms: duration of each beep
// ─────────────────────────────────────────────────────────────────────────────
void playBeeps(int count, int freq_hz = 880, int dur_ms = 150) {
  int totalSamples = (SAMPLE_RATE * dur_ms) / 1000;
  static uint16_t tone[CHUNK_SAMPLES * 2];

  for (int b = 0; b < count; b++) {
    int samplesLeft = totalSamples;
    int phase = 0;
    while (samplesLeft > 0) {
      int n = min(samplesLeft, CHUNK_SAMPLES);
      for (int i = 0; i < n; i++) {
        float s = sinf(2.0f * M_PI * freq_hz * phase / SAMPLE_RATE);
        uint16_t v = (uint16_t)((int)(s * 20000) + 0x8000);  // signed → unsigned DAC
        tone[i * 2]     = v;
        tone[i * 2 + 1] = v;
        phase++;
      }
      size_t written;
      i2s_write(I2S_DAC, tone, n * 4, &written, portMAX_DELAY);
      samplesLeft -= n;
    }
    // Gap between beeps
    if (b < count - 1) {
      int gapSamples = SAMPLE_RATE / 10;  // 100 ms gap
      while (gapSamples > 0) {
        int n = min(gapSamples, CHUNK_SAMPLES);
        memset(tone, 0x80, n * 4);  // mid-scale silence (DAC)
        size_t written;
        i2s_write(I2S_DAC, tone, n * 4, &written, portMAX_DELAY);
        gapSamples -= n;
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
static float computeRms(const int16_t* buf, int n) {
  int64_t sum = 0;
  for (int i = 0; i < n; i++) { int32_t v = buf[i]; sum += (int64_t)v * v; }
  return (n > 0) ? sqrtf((float)sum / n) : 0.0f;
}

static void sendFrame(const int16_t* pcm, int samples) {
  uint16_t len = samples * 2;
  txBuf[0] = 0xAA; txBuf[1] = 0x55;
  txBuf[2] = (len >> 8) & 0xFF; txBuf[3] = len & 0xFF;
  memcpy(txBuf + 4, pcm, len);
  udpMic.beginPacket(SERVER_IP, MIC_PORT);
  udpMic.write(txBuf, 4 + len);
  udpMic.endPacket();
  totalPktsSent++;
}

static void sendEOS() {
  uint8_t eos[4] = {0xAA, 0x55, 0x00, 0x00};
  udpMic.beginPacket(SERVER_IP, MIC_PORT);
  udpMic.write(eos, 4);
  udpMic.endPacket();
  Serial.printf("[MIC] EOS sent — %u packets\n", totalPktsSent);
}

void goToSleep() {
  Serial.println("[SLEEP] Deep sleep → wakes on GPIO35 HIGH");
  Serial.flush();
  delay(200);
  i2s_stop(I2S_MIC);  i2s_driver_uninstall(I2S_MIC);
  i2s_stop(I2S_DAC);  i2s_driver_uninstall(I2S_DAC);
  WiFi.disconnect(true);
  WiFi.mode(WIFI_OFF);
  delay(100);
  esp_sleep_enable_ext0_wakeup(WAKE_PIN, 1);
  esp_deep_sleep_start();
}

void sleepWithBeeps(int count, const char* reason) {
  Serial.printf("[SLEEP] %s — beeping %d times then sleeping\n", reason, count);
  playBeeps(count, 440, 200);   // low tone = warning
  delay(300);
  goToSleep();
}

// ─────────────────────────────────────────────────────────────────────────────
// I2S setup
// ─────────────────────────────────────────────────────────────────────────────
void setupMic() {
  const i2s_config_t cfg = {
    .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate          = SAMPLE_RATE,
    .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count        = 8,
    .dma_buf_len          = CHUNK_SAMPLES,
    .use_apll             = true,
    .tx_desc_auto_clear   = false,
    .fixed_mclk           = 0
  };
  const i2s_pin_config_t pins = {
    .bck_io_num   = MIC_SCK,
    .ws_io_num    = MIC_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num  = MIC_SD
  };
  ESP_ERROR_CHECK(i2s_driver_install(I2S_MIC, &cfg, 0, NULL));
  ESP_ERROR_CHECK(i2s_set_pin(I2S_MIC, &pins));
  i2s_zero_dma_buffer(I2S_MIC);
  Serial.println("[MIC] Ready  SCK=14 WS=15 SD=32");
}

void setupDac() {
  const i2s_config_t cfg = {
    .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX | I2S_MODE_DAC_BUILT_IN),
    .sample_rate          = SAMPLE_RATE,
    .bits_per_sample      = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format       = I2S_CHANNEL_FMT_RIGHT_LEFT,
    .communication_format = (i2s_comm_format_t)I2S_COMM_FORMAT_STAND_MSB,
    .intr_alloc_flags     = 0,
    .dma_buf_count        = 8,
    .dma_buf_len          = CHUNK_SAMPLES,
    .use_apll             = true,
    .tx_desc_auto_clear   = true
  };
  ESP_ERROR_CHECK(i2s_driver_install(I2S_DAC, &cfg, 0, NULL));
  ESP_ERROR_CHECK(i2s_set_dac_mode(I2S_DAC_CHANNEL_LEFT_EN));
  i2s_zero_dma_buffer(I2S_DAC);
  Serial.println("[DAC] Ready  GPIO25");
}

// ══════════════════════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("\n╔══════════════════════════════════════╗");
  Serial.println("║    ESP32 Gemini Voice Assistant  v3  ║");
  Serial.println("╚══════════════════════════════════════╝");

  esp_sleep_wakeup_cause_t cause = esp_sleep_get_wakeup_cause();

  if (cause == ESP_SLEEP_WAKEUP_EXT0) {
    Serial.println("[WAKE] GPIO35 triggered — sound detected");
  } else {
    // Cold boot → arm wakeup and sleep immediately
    Serial.println("[BOOT] Cold boot → arming GPIO35 wakeup");
    Serial.println("[BOOT] Make a sound near KY-037 to start the assistant");
    Serial.flush();
    esp_sleep_enable_ext0_wakeup(WAKE_PIN, 1);
    delay(50);
    esp_deep_sleep_start();
    // never reaches here
  }

  // ── GPIO35 confirmed HIGH — run the pipeline ──────────────────────────────
  setupMic();
  setupDac();

  // Single confirming beep to signal wakeup
  playBeeps(1, 1200, 100);

  // ── WiFi ──────────────────────────────────────────────────────────────────
  WiFi.mode(WIFI_STA);
  WiFi.begin(SSID, PASSWORD);
  Serial.print("[WiFi] Connecting");
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
    if (++tries > 40) {
      Serial.println("\n[WiFi] Failed");
      sleepWithBeeps(3, "WiFi failed");
    }
  }
  Serial.printf("\n[WiFi] %s  RSSI=%d dBm → %s:%d\n",
                WiFi.localIP().toString().c_str(), WiFi.RSSI(), SERVER_IP, MIC_PORT);

  udpMic.begin(MIC_PORT);
  udpSpk.begin(SPK_PORT);

  // Flush DMA warm-up
  size_t dummy;
  for (int i = 0; i < 8; i++)
    i2s_read(I2S_MIC, micRaw, sizeof(micRaw), &dummy, portMAX_DELAY);

  streamStartMs  = millis();
  lastVoiceMs    = millis();
  totalPktsSent  = 0;
  speechDetected = false;
  state          = STREAMING;

  Serial.printf("[STREAM] Started  VAD=%d  silence=%dms  max=%dms\n",
                VAD_SPEECH_RMS, SILENCE_TIMEOUT_MS, MAX_STREAM_MS);
}

// ══════════════════════════════════════════════════════════════════════════════
void loop() {

  // ── STREAMING ──────────────────────────────────────────────────────────────
  if (state == STREAMING) {
    size_t bytesRead = 0;
    i2s_read(I2S_MIC, micRaw, sizeof(micRaw), &bytesRead, pdMS_TO_TICKS(30));

    if (bytesRead > 0) {
      int n = bytesRead / 4;

      // FIX: shift >>14 not >>11 — INMP441 data is in upper 24 bits of 32-bit frame.
      // >>14 maps the 24-bit signed value into a clean 16-bit range without clipping.
      for (int i = 0; i < n; i++) {
        int32_t s = micRaw[i] >> 14;
        micPcm[i] = (int16_t)constrain(s, -32768, 32767);
      }

      float rms = computeRms(micPcm, n);

      static int rmsLog = 0;
      if (++rmsLog >= 20) {
        Serial.printf("[VAD] RMS=%.0f  threshold=%d  speech=%s\n",
                      rms, VAD_SPEECH_RMS, speechDetected ? "YES" : "no");
        rmsLog = 0;
      }

      if (rms > VAD_SPEECH_RMS) {
        lastVoiceMs    = millis();
        speechDetected = true;
      }

      sendFrame(micPcm, n);
    }

    bool silenceTimeout = speechDetected && (millis() - lastVoiceMs) > SILENCE_TIMEOUT_MS;
    bool maxDuration    = (millis() - streamStartMs) >= MAX_STREAM_MS;

    if (silenceTimeout) {
      Serial.printf("[VAD] Silence %.1f s → EOS\n", (millis()-lastVoiceMs)/1000.0f);
      sendEOS();
      state         = WAIT_REPLY;
      waitReplyStart = millis();
    } else if (maxDuration) {
      if (!speechDetected) {
        // No voice detected at all within 8 s — no point waiting for server
        Serial.println("[VAD] No speech in 8 s → sleeping");
        sleepWithBeeps(2, "no speech detected");
      }
      Serial.println("[VAD] Hard cap → EOS");
      sendEOS();
      state          = WAIT_REPLY;
      waitReplyStart = millis();
    }
  }

  // ── WAIT_REPLY / PLAYING ───────────────────────────────────────────────────
  else {

    // ── Server reply timeout ────────────────────────────────────────────────
    if (state == WAIT_REPLY && (millis() - waitReplyStart) > SERVER_REPLY_TIMEOUT_MS) {
      Serial.println("[WAIT] Server did not reply in time");
      sleepWithBeeps(3, "server timeout");
    }

    int pktSize = udpSpk.parsePacket();

    if (pktSize >= 4) {
      int n = udpSpk.read(rxBuf, sizeof(rxBuf));
      if (n < 4 || rxBuf[0] != 0xAA || rxBuf[1] != 0x55) return;

      uint16_t payloadLen = ((uint16_t)rxBuf[2] << 8) | rxBuf[3];

      if (payloadLen == 0) {
        // EOS from server
        if (state == WAIT_REPLY) {
          // Server sent EOS without sending any audio (clip too short etc.)
          Serial.println("[PLAY] Server EOS with no audio → sleeping");
          sleepWithBeeps(2, "no audio from server");
        }
        Serial.println("[PLAY] Playback complete → sleeping");
        delay(300);
        goToSleep();
        return;
      }

      if (state == WAIT_REPLY) {
        Serial.println("[PLAY] TTS arriving — playing");
        state     = PLAYING;
        lastPktMs = millis();
      }
      lastPktMs = millis();

      int16_t* samples    = (int16_t*)(rxBuf + 4);
      int      numSamples = payloadLen / 2;
      for (int i = 0; i < numSamples && i < CHUNK_SAMPLES; i++) {
        uint16_t v        = (uint16_t)((int32_t)samples[i] + 0x8000);
        dacBuf[i * 2]     = v;
        dacBuf[i * 2 + 1] = v;
      }
      size_t written;
      i2s_write(I2S_DAC, dacBuf, numSamples * 4, &written, portMAX_DELAY);

    } else if (state == PLAYING && (millis() - lastPktMs) > PLAY_DONE_TIMEOUT_MS) {
      Serial.println("[PLAY] Playback done → sleeping");
      delay(300);
      goToSleep();
    }
  }
}
