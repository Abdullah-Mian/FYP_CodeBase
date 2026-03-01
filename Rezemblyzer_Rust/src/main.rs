cat << 'RUSTEOF' > src/main.rs
use anyhow::Result;
use ndarray::{Array, Array1, Array2, Ix2};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use std::io::{self, Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

// ── UART Configuration (must match ESP32) ─────────────────────────────────────
const SERIAL_PORT_PATH: &str = "/dev/serial0";
const BAUD_RATE: u32 = 921_600;
const SAMPLE_RATE: u32 = 16_000;
const RECORD_SECONDS: usize = 3;
const SAMPLES_NEEDED: usize = SAMPLE_RATE as usize * RECORD_SECONDS;
const BYTES_NEEDED: usize = SAMPLES_NEEDED * 2; // 16-bit PCM = 2 bytes per sample

// ── Frame protocol ────────────────────────────────────────────────────────────
const SYNC_BYTE_1: u8 = 0xAA;
const SYNC_BYTE_2: u8 = 0x55;

// ── Voice auth ────────────────────────────────────────────────────────────────
const THRESHOLD: f32 = 0.75;
const ENROLLMENT_FILE: &str = "voice_profile.bin";

fn main() -> Result<()> {
    ort::init().commit();

    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file("voice_auth.onnx")?;
    let model = Arc::new(Mutex::new(model));

    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    // ── Signal handling ───────────────────────────────────────────────────────
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        println!("\n[VS] Shutting down...");
        r.store(false, Ordering::SeqCst);
    })?;

    match mode {
        "enroll" => enroll(&model, &running)?,
        "verify" => verify(&model, &running)?,
        "listen" => listen(&model, &running)?,
        _ => {
            println!("╔══════════════════════════════════════╗");
            println!("║       VOICE SENTRY v0.2.0            ║");
            println!("║   INMP441 → ESP32 → UART → Pi       ║");
            println!("╠══════════════════════════════════════╣");
            println!("║  Usage:                              ║");
            println!("║    ./voice_sentry enroll  - Record   ║");
            println!("║         your voice profile           ║");
            println!("║    ./voice_sentry verify  - One      ║");
            println!("║         time voice check             ║");
            println!("║    ./voice_sentry listen  - Live     ║");
            println!("║         continuous auth              ║");
            println!("╚══════════════════════════════════════╝");
        }
    }

    Ok(())
}

// ── Serial helpers ────────────────────────────────────────────────────────────

fn open_serial() -> Result<Box<dyn serialport::SerialPort>> {
    let port = serialport::new(SERIAL_PORT_PATH, BAUD_RATE)
        .data_bits(serialport::DataBits::Eight)
        .parity(serialport::Parity::None)
        .stop_bits(serialport::StopBits::One)
        .timeout(Duration::from_millis(3000))
        .open()?;
    println!("[VS] Serial: {} @ {} baud", SERIAL_PORT_PATH, BAUD_RATE);
    Ok(port)
}

fn read_byte(port: &mut Box<dyn serialport::SerialPort>) -> io::Result<Option<u8>> {
    let mut buf = [0u8; 1];
    match port.read(&mut buf) {
        Ok(n) if n > 0 => Ok(Some(buf[0])),
        Ok(_) => Ok(None),
        Err(e) if e.kind() == io::ErrorKind::TimedOut => Ok(None),
        Err(e) => Err(e),
    }
}

fn read_exact_n(port: &mut Box<dyn serialport::SerialPort>, n: usize) -> io::Result<Option<Vec<u8>>> {
    let mut buf = vec![0u8; n];
    let mut total = 0;
    while total < n {
        match port.read(&mut buf[total..]) {
            Ok(0) => return Ok(None),
            Ok(k) => total += k,
            Err(e) if e.kind() == io::ErrorKind::TimedOut => return Ok(None),
            Err(e) => return Err(e),
        }
    }
    Ok(Some(buf))
}

fn find_sync(port: &mut Box<dyn serialport::SerialPort>) -> io::Result<bool> {
    let mut state: u8 = 0;
    loop {
        match read_byte(port)? {
            None => return Ok(false),
            Some(b) => match state {
                0 => { if b == SYNC_BYTE_1 { state = 1; } }
                1 => {
                    if b == SYNC_BYTE_2 { return Ok(true); }
                    else if b == SYNC_BYTE_1 { state = 1; }
                    else { state = 0; }
                }
                _ => state = 0,
            },
        }
    }
}

fn read_frame(port: &mut Box<dyn serialport::SerialPort>) -> io::Result<Option<Vec<u8>>> {
    if !find_sync(port)? { return Ok(None); }
    let len_bytes = match read_exact_n(port, 2)? {
        Some(b) => b,
        None => return Ok(None),
    };
    let frame_len = ((len_bytes[0] as usize) << 8) | len_bytes[1] as usize;
    if frame_len == 0 { return Ok(Some(vec![])); }
    read_exact_n(port, frame_len)
}

/// Collect PCM bytes from UART frames until we have enough for `seconds` of audio.
fn record_from_uart(
    port: &mut Box<dyn serialport::SerialPort>,
    running: &Arc<AtomicBool>,
) -> Result<Vec<f32>> {
    let mut pcm_bytes: Vec<u8> = Vec::with_capacity(BYTES_NEEDED);

    println!("[VS] Speak now! Recording {}s of audio via UART...", RECORD_SECONDS);

    while pcm_bytes.len() < BYTES_NEEDED && running.load(Ordering::SeqCst) {
        match read_frame(port) {
            Ok(None) => {
                // Timeout — no data yet
                continue;
            }
            Ok(Some(data)) if data.is_empty() => {
                // End-of-stream from ESP32
                if !pcm_bytes.is_empty() {
                    println!("[VS] ESP32 ended stream early ({} bytes captured)", pcm_bytes.len());
                    break;
                }
            }
            Ok(Some(data)) => {
                pcm_bytes.extend_from_slice(&data);
                let duration = pcm_bytes.len() as f64 / (SAMPLE_RATE as f64 * 2.0);
                print!("\r[VS] Recording: {:.1}s / {}s   ", duration, RECORD_SECONDS);
                io::stdout().flush()?;
            }
            Err(e) => {
                return Err(anyhow::anyhow!("Serial error: {}", e));
            }
        }
    }
    println!("\r[VS] ✓ Recording complete! ({} bytes)       ", pcm_bytes.len());

    // Convert 16-bit LE PCM bytes to f32 samples normalized to [-1.0, 1.0]
    let samples: Vec<f32> = pcm_bytes
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    Ok(samples)
}

// ── ML helpers ────────────────────────────────────────────────────────────────

fn get_embedding(model: &Arc<Mutex<Session>>, samples: Vec<f32>) -> Result<Array1<f32>> {
    let len = samples.len();
    let input: Array<f32, Ix2> = Array2::from_shape_vec((1, len), samples)
        .map_err(|e| anyhow::anyhow!("Shape error: {:?}", e))?;

    let input_value = Value::from_array(input)?;
    let mut model_guard = model.lock().unwrap();
    let outputs = model_guard.run(ort::inputs!["audio" => input_value])?;

    let (shape, data) = outputs["embedding"].try_extract_tensor::<f32>()?;
    let dim_size = shape[1] as usize;
    let embedding = Array1::from_shape_vec(dim_size, data.to_vec())
        .map_err(|e| anyhow::anyhow!("Shape error: {:?}", e))?;

    Ok(embedding)
}

fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
    dot / (norm_a * norm_b)
}

fn save_embedding(embedding: &Array1<f32>) -> Result<()> {
    let bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(ENROLLMENT_FILE, bytes)?;
    Ok(())
}

fn load_embedding() -> Result<Array1<f32>> {
    let bytes = std::fs::read(ENROLLMENT_FILE)?;
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Ok(Array1::from_vec(floats))
}

// ── Modes ─────────────────────────────────────────────────────────────────────

fn enroll(model: &Arc<Mutex<Session>>, running: &Arc<AtomicBool>) -> Result<()> {
    println!("\n🎤 VOICE ENROLLMENT (via INMP441 → ESP32 → UART)");
    println!("  You will record 3 samples to create your voice profile.\n");

    let mut port = open_serial()?;
    port.clear(serialport::ClearBuffer::Input).ok();

    let mut embeddings: Vec<Array1<f32>> = Vec::new();

    for i in 1..=3 {
        println!("\n  Sample {}/3 — trigger ESP32 wake word, then speak for {}s", i, RECORD_SECONDS);
        let samples = record_from_uart(&mut port, running)?;

        if samples.is_empty() {
            println!("  ⚠ No audio captured, try again");
            continue;
        }

        let embedding = get_embedding(model, samples)?;
        embeddings.push(embedding);
    }

    if embeddings.is_empty() {
        return Err(anyhow::anyhow!("No samples captured. Check ESP32 connection."));
    }

    // Average & normalize
    let dim = embeddings[0].len();
    let mut avg = Array1::<f32>::zeros(dim);
    for emb in &embeddings {
        avg = avg + emb;
    }
    avg /= embeddings.len() as f32;
    let norm = avg.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 { avg /= norm; }

    save_embedding(&avg)?;
    println!("\n  ✅ Voice profile saved to '{}'", ENROLLMENT_FILE);
    println!("  Run './voice_sentry verify' to test authentication.\n");
    Ok(())
}

fn verify(model: &Arc<Mutex<Session>>, running: &Arc<AtomicBool>) -> Result<()> {
    let enrolled = load_embedding()
        .map_err(|_| anyhow::anyhow!("No voice profile found. Run './voice_sentry enroll' first!"))?;

    println!("\n🔐 VOICE VERIFICATION (via INMP441 → ESP32 → UART)");
    println!("  Trigger ESP32 and speak for {}s.\n", RECORD_SECONDS);

    let mut port = open_serial()?;
    port.clear(serialport::ClearBuffer::Input).ok();

    let samples = record_from_uart(&mut port, running)?;
    let embedding = get_embedding(model, samples)?;
    let similarity = cosine_similarity(&enrolled, &embedding);

    println!("\n  Similarity: {:.4}", similarity);
    println!("  Threshold:  {:.4}", THRESHOLD);
    if similarity >= THRESHOLD {
        println!("  ✅ AUTHORIZED — Voice match confirmed!\n");
    } else {
        println!("  ❌ DENIED — Voice does not match!\n");
    }

    Ok(())
}

fn listen(model: &Arc<Mutex<Session>>, running: &Arc<AtomicBool>) -> Result<()> {
    let enrolled = load_embedding()
        .map_err(|_| anyhow::anyhow!("No voice profile found. Run './voice_sentry enroll' first!"))?;

    println!("\n👂 CONTINUOUS LISTENING MODE (via INMP441 → ESP32 → UART)");
    println!("  Monitoring for authorized voice... (Ctrl+C to stop)\n");

    let mut port = open_serial()?;
    port.clear(serialport::ClearBuffer::Input).ok();

    while running.load(Ordering::SeqCst) {
        let samples = record_from_uart(&mut port, running)?;
        if samples.is_empty() { continue; }

        let embedding = get_embedding(model, samples)?;
        let similarity = cosine_similarity(&enrolled, &embedding);

        if similarity >= THRESHOLD {
            println!("  ✅ Authorized (similarity: {:.4})", similarity);
        } else {
            println!("  ❌ Unauthorized (similarity: {:.4})", similarity);
        }
    }

    Ok(())
}
RUSTEOF