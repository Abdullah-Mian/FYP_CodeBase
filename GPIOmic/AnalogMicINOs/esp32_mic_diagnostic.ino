/*
 * ESP32 Microphone Diagnostic & Calibration Tool
 * ================================================
 * Flash this, open Serial Monitor at 115200 baud.
 * It will:
 *   1. Measure DC offset (bias voltage)
 *   2. Measure noise floor (silence RMS)
 *   3. Measure speech amplitude (RMS when you talk)
 *   4. Print recommended thresholds
 *   5. Test the D0 digital pin behavior
 *
 * Wiring:
 *   Mic AO  -> GPIO36 (VP)
 *   Mic D0  -> GPIO33
 *   Mic +   -> 3.3V
 *   Mic GND -> GND
 *
 * Board: ESP32 Dev Module, Arduino IDE 2.3.7
 */

#define MIC_ANALOG_PIN  36    // AO pin -> GPIO36 (ADC1_CH0)
#define MIC_DIGITAL_PIN 33    // D0 pin -> GPIO33

#define NUM_SAMPLES     4000  // Samples per measurement window
#define SAMPLE_DELAY_US 125   // 8kHz sampling for diagnostics

// Measurement results
float dc_offset = 0;
float noise_rms = 0;
float speech_rms = 0;
float noise_peak = 0;
float speech_peak = 0;

void collect_samples(int16_t* buffer, int count) {
  for (int i = 0; i < count; i++) {
    buffer[i] = analogRead(MIC_ANALOG_PIN);
    delayMicroseconds(SAMPLE_DELAY_US);
  }
}

void analyze_buffer(int16_t* buffer, int count, float* out_dc, float* out_rms, float* out_peak) {
  // Pass 1: DC offset
  int64_t sum = 0;
  for (int i = 0; i < count; i++) {
    sum += buffer[i];
  }
  *out_dc = (float)sum / count;

  // Pass 2: RMS and peak relative to DC
  float sum_sq = 0;
  float max_abs = 0;
  for (int i = 0; i < count; i++) {
    float val = (float)buffer[i] - *out_dc;
    sum_sq += val * val;
    float a = fabs(val);
    if (a > max_abs) max_abs = a;
  }
  *out_rms = sqrt(sum_sq / count);
  *out_peak = max_abs;
}

void print_separator() {
  Serial.println("============================================================");
}

void setup() {
  Serial.begin(115200);
  delay(2000); // Wait for serial monitor to connect

  analogReadResolution(12);       // 12-bit ADC (0-4095)
  analogSetAttenuation(ADC_11db); // Full 0-3.3V range
  pinMode(MIC_ANALOG_PIN, INPUT);
  pinMode(MIC_DIGITAL_PIN, INPUT);

  print_separator();
  Serial.println("  ESP32 MICROPHONE DIAGNOSTIC TOOL");
  Serial.println("  KY-037/KY-038 Calibration");
  print_separator();
  Serial.println();
  Serial.printf("ADC Pin: GPIO%d (12-bit, 0-4095)\n", MIC_ANALOG_PIN);
  Serial.printf("D0 Pin:  GPIO%d\n", MIC_DIGITAL_PIN);
  Serial.printf("Samples per window: %d\n", NUM_SAMPLES);
  Serial.printf("Sample rate: ~%d Hz\n", 1000000 / SAMPLE_DELAY_US);
  Serial.println();

  // ==================== PHASE 1: SILENCE ====================
  Serial.println(">>> PHASE 1: SILENCE MEASUREMENT");
  Serial.println("    Stay QUIET for 5 seconds...");
  Serial.println();
  delay(2000);

  int16_t* buffer = (int16_t*)malloc(NUM_SAMPLES * sizeof(int16_t));
  if (!buffer) {
    Serial.println("ERROR: Could not allocate buffer!");
    while (1) delay(1000);
  }

  // Take 3 silence measurements and average
  float total_dc = 0, total_rms = 0, total_peak = 0;
  for (int m = 0; m < 3; m++) {
    float dc, rms, peak;
    collect_samples(buffer, NUM_SAMPLES);
    analyze_buffer(buffer, NUM_SAMPLES, &dc, &rms, &peak);
    total_dc += dc;
    total_rms += rms;
    total_peak += peak;
    Serial.printf("  Silence measurement %d: DC=%.1f  RMS=%.2f  Peak=%.1f\n", m + 1, dc, rms, peak);
    delay(500);
  }
  dc_offset = total_dc / 3.0;
  noise_rms = total_rms / 3.0;
  noise_peak = total_peak / 3.0;

  Serial.println();
  Serial.printf("  >> Silence DC Offset: %.1f (raw ADC)\n", dc_offset);
  Serial.printf("  >> Silence DC Voltage: %.3f V\n", dc_offset * 3.3 / 4095.0);
  Serial.printf("  >> Silence RMS Noise:  %.2f counts\n", noise_rms);
  Serial.printf("  >> Silence Peak Noise: %.1f counts\n", noise_peak);
  Serial.println();

  // ==================== PHASE 2: SPEECH ====================
  print_separator();
  Serial.println(">>> PHASE 2: SPEECH MEASUREMENT");
  Serial.println("    SPEAK NORMALLY into the mic for 8 seconds.");
  Serial.println("    Try different distances: 5cm, 15cm, 30cm, 60cm");
  Serial.println("    Starting in 3 seconds...");
  delay(3000);
  Serial.println("    >> RECORDING NOW - SPEAK! <<");
  Serial.println();

  float max_speech_rms = 0;
  float max_speech_peak = 0;

  for (int m = 0; m < 8; m++) {
    float dc, rms, peak;
    collect_samples(buffer, NUM_SAMPLES);
    analyze_buffer(buffer, NUM_SAMPLES, &dc, &rms, &peak);

    // Read D0 state
    int d0_state = digitalRead(MIC_DIGITAL_PIN);

    Serial.printf("  Window %d: DC=%.1f  RMS=%.2f  Peak=%.1f  D0=%s\n",
                  m + 1, dc, rms, peak, d0_state ? "HIGH" : "LOW");

    if (rms > max_speech_rms) max_speech_rms = rms;
    if (peak > max_speech_peak) max_speech_peak = peak;
  }
  speech_rms = max_speech_rms;
  speech_peak = max_speech_peak;

  Serial.println();
  Serial.printf("  >> Max Speech RMS:  %.2f counts\n", speech_rms);
  Serial.printf("  >> Max Speech Peak: %.1f counts\n", speech_peak);
  Serial.println();

  // ==================== PHASE 3: CLAP TEST ====================
  print_separator();
  Serial.println(">>> PHASE 3: CLAP / LOUD SOUND TEST");
  Serial.println("    CLAP your hands near the mic several times.");
  Serial.println("    Starting in 2 seconds...");
  delay(2000);
  Serial.println("    >> CLAP NOW! <<");

  float max_clap_rms = 0;
  float max_clap_peak = 0;

  for (int m = 0; m < 5; m++) {
    float dc, rms, peak;
    collect_samples(buffer, NUM_SAMPLES);
    analyze_buffer(buffer, NUM_SAMPLES, &dc, &rms, &peak);

    int d0_state = digitalRead(MIC_DIGITAL_PIN);

    Serial.printf("  Window %d: DC=%.1f  RMS=%.2f  Peak=%.1f  D0=%s\n",
                  m + 1, dc, rms, peak, d0_state ? "HIGH" : "LOW");

    if (rms > max_clap_rms) max_clap_rms = rms;
    if (peak > max_clap_peak) max_clap_peak = peak;
  }

  Serial.println();

  // ==================== PHASE 4: D0 RESPONSIVENESS ====================
  print_separator();
  Serial.println(">>> PHASE 4: D0 PIN MONITOR (10 seconds)");
  Serial.println("    Make sounds - watching D0 transitions...");
  Serial.println();

  int d0_trigger_count = 0;
  int last_d0 = digitalRead(MIC_DIGITAL_PIN);
  unsigned long start = millis();
  unsigned long last_print = 0;

  while (millis() - start < 10000) {
    int d0 = digitalRead(MIC_DIGITAL_PIN);
    if (d0 != last_d0) {
      d0_trigger_count++;
      if (d0 == HIGH) {
        Serial.printf("  D0 -> HIGH at %lu ms\n", millis() - start);
      }
      last_d0 = d0;
    }
    delayMicroseconds(100);
  }

  Serial.printf("  >> D0 transitions in 10s: %d\n", d0_trigger_count);
  Serial.println();

  // ==================== PHASE 5: LIVE STREAM (for graph) ====================
  print_separator();
  Serial.println(">>> PHASE 5: LIVE AUDIO STREAM (30 seconds)");
  Serial.println("    Open Serial Plotter to see waveform.");
  Serial.println("    Format: CENTERED_VALUE,RMS_ENVELOPE,THRESHOLD");
  Serial.println();

  // Calculate recommended thresholds
  float recommended_vad_threshold = noise_rms * 3.0; // 3x noise floor
  if (recommended_vad_threshold < 5.0) recommended_vad_threshold = 5.0;

  float snr = 0;
  if (noise_rms > 0) snr = speech_rms / noise_rms;

  print_separator();
  Serial.println("============ CALIBRATION RESULTS ============");
  print_separator();
  Serial.printf("DC Offset (raw):        %.1f\n", dc_offset);
  Serial.printf("DC Offset (voltage):    %.3f V\n", dc_offset * 3.3 / 4095.0);
  Serial.printf("Noise Floor RMS:        %.2f counts\n", noise_rms);
  Serial.printf("Noise Floor Peak:       %.1f counts\n", noise_peak);
  Serial.printf("Speech RMS (max):       %.2f counts\n", speech_rms);
  Serial.printf("Speech Peak (max):      %.1f counts\n", speech_peak);
  Serial.printf("Clap RMS (max):         %.2f counts\n", max_clap_rms);
  Serial.printf("Clap Peak (max):        %.1f counts\n", max_clap_peak);
  Serial.printf("SNR (speech/noise):     %.1f : 1\n", snr);
  Serial.printf("D0 transitions (10s):   %d\n", d0_trigger_count);
  Serial.println();
  Serial.printf(">> Recommended VAD threshold: %.1f (RMS)\n", recommended_vad_threshold);
  Serial.printf(">> Signal quality grade: ");

  if (snr >= 10) {
    Serial.println("EXCELLENT - Good for voice recognition");
  } else if (snr >= 5) {
    Serial.println("GOOD - Usable for close-range voice");
  } else if (snr >= 2.5) {
    Serial.println("FAIR - May work close up, noisy environment will fail");
  } else {
    Serial.println("POOR - Consider upgrading to INMP441 or MAX9814");
  }

  Serial.println();
  Serial.println(">> COPY EVERYTHING ABOVE AND SHARE IT <<");
  print_separator();

  // Live stream with envelope for Serial Plotter
  Serial.println();
  Serial.println("Starting live stream (Serial Plotter mode)...");
  delay(2000);

  // Running RMS calculation
  const int rms_window = 64;
  float rms_buf[rms_window];
  int rms_idx = 0;
  for (int i = 0; i < rms_window; i++) rms_buf[i] = 0;

  unsigned long stream_start = millis();
  while (millis() - stream_start < 30000) {
    int raw = analogRead(MIC_ANALOG_PIN);
    float centered = (float)raw - dc_offset;

    // Running RMS
    rms_buf[rms_idx] = centered * centered;
    rms_idx = (rms_idx + 1) % rms_window;
    float sum_sq2 = 0;
    for (int i = 0; i < rms_window; i++) sum_sq2 += rms_buf[i];
    float running_rms = sqrt(sum_sq2 / rms_window);

    // Print: centered, running_rms, threshold (for Serial Plotter)
    Serial.printf("%.0f,%.1f,%.1f\n", centered, running_rms, recommended_vad_threshold);

    delayMicroseconds(SAMPLE_DELAY_US);
  }

  free(buffer);

  Serial.println();
  Serial.println("Diagnostic complete. Copy the CALIBRATION RESULTS section above.");
}

void loop() {
  // Nothing - diagnostic runs once in setup()
  delay(10000);
}