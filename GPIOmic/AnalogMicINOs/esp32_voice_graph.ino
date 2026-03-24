/*
 * ESP32 Voice Input Graph
 * Reads analog audio from KY-037/038 AO pin via ADC
 * and prints values for Arduino Serial Plotter.
 *
 * Board: ESP32 Dev Module (NodeMCU-32S)
 * Arduino IDE 2.3.7+
 *
 * Wiring:
 *   Mic AO  -> GPIO36 (VP)
 *   Mic +   -> 3.3V
 *   Mic GND -> GND
 */

#define MIC_ANALOG_PIN 36  // ADC1_CH0, GPIO36 (VP)
#define SAMPLE_RATE_US 125 // ~8000 samples/sec (125 µs per sample)

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);       // 12-bit ADC (0–4095)
  analogSetAttenuation(ADC_11db); // Full 0–3.3V range
  pinMode(MIC_ANALOG_PIN, INPUT);
  Serial.println("ESP32 Voice Graph Ready");
}

void loop() {
  int sample = analogRead(MIC_ANALOG_PIN);

  // Center around zero for waveform display (2048 is DC midpoint)
  int centered = sample - 2048;

  Serial.println(centered);

  delayMicroseconds(SAMPLE_RATE_US);
}