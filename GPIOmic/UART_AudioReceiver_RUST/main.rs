// ============================================================================
//  rpi4_audio_receiver – Rust port of rpi4_audio_receiver.py
// ============================================================================
//  Receives framed 16-bit PCM audio over UART at 921600 baud from an ESP32
//  and saves each voice session as a timestamped WAV file.
//
//  Frame protocol  [0xAA][0x55][LEN_H][LEN_L][PCM_DATA …]
//                  LEN == 0  ⟹  end-of-stream marker
//
//  Wiring:
//      ESP32 GPIO17 (TX2)  →  RPi4 GPIO15 (RXD, Pin 10)
//      ESP32 GND           →  RPi4 GND
//
//  Dependencies: serialport 4, ctrlc 3  (both pure-Rust, no C libs required)
// ============================================================================

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serialport::SerialPort;

// ── Configuration ─────────────────────────────────────────────────────────────

const SERIAL_PORT_PATH: &str = "/dev/serial0";
const BAUD_RATE: u32 = 921_600;
const SAMPLE_RATE: u32 = 16_000;
const CHANNELS: u16 = 1;
const SAMPLE_WIDTH: u16 = 2; // bytes per sample  (16-bit PCM)
const RECEIVE_TIMEOUT_MS: u64 = 3_000; // 3 s – mirrors Python timeout=3.0

const SYNC_BYTE_1: u8 = 0xAA;
const SYNC_BYTE_2: u8 = 0x55;

// ── Output directory ──────────────────────────────────────────────────────────

fn output_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    PathBuf::from(home).join("audio_recordings")
}

fn ensure_output_dir(dir: &Path) -> io::Result<()> {
    fs::create_dir_all(dir)
}

// ── Timestamp helpers (UTC, stdlib only – no chrono dependency) ───────────────

/// Convert a Unix timestamp (seconds) to (year, month, day) using the
/// proleptic Gregorian calendar – Howard Hinnant's "civil from days" algorithm.
fn days_to_ymd(days: i64) -> (i64, i64, i64) {
    let z   = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;                              // [0, 146096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y   = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);       // [0, 365]
    let mp  = (5 * doy + 2) / 153;                            // [0, 11]
    let d   = doy - (153 * mp + 2) / 5 + 1;                  // [1, 31]
    let m   = if mp < 10 { mp + 3 } else { mp - 9 };         // [1, 12]
    let y   = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Return current UTC time formatted as `"YYYYMMDD_HHMMSS"`.
/// Matches `datetime.now().strftime("%Y%m%d_%H%M%S")` for UTC clocks.
fn utc_timestamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let days       = secs / 86_400;
    let time_secs  = secs % 86_400;
    let hh         = time_secs / 3_600;
    let mm         = (time_secs % 3_600) / 60;
    let ss         = time_secs % 60;

    let (y, mo, d) = days_to_ymd(days);
    format!("{:04}{:02}{:02}_{:02}{:02}{:02}", y, mo, d, hh, mm, ss)
}

/// `"HH:MM:SS"` slice from a timestamp produced by `utc_timestamp()`.
fn time_from_timestamp(ts: &str) -> String {
    // ts is "YYYYMMDD_HHMMSS"
    let t = &ts[9..]; // "HHMMSS"
    format!("{}:{}:{}", &t[0..2], &t[2..4], &t[4..6])
}

fn generate_filename(dir: &Path) -> PathBuf {
    dir.join(format!("recording_{}.wav", utc_timestamp()))
}

// ── WAV writer (no external crate – written to spec) ─────────────────────────

/// Write `pcm_data` (raw 16-bit LE PCM) as a standards-compliant WAV file.
/// Mirrors `save_wav()` in the Python original.
fn save_wav(filename: &Path, pcm_data: &[u8]) -> io::Result<()> {
    let data_size   = pcm_data.len() as u32;
    let byte_rate   = SAMPLE_RATE * CHANNELS as u32 * SAMPLE_WIDTH as u32;
    let block_align = CHANNELS * SAMPLE_WIDTH;
    let bits        = SAMPLE_WIDTH * 8;

    let mut f = File::create(filename)?;

    // ── RIFF chunk ───────────────────────────────────────────────────────────
    f.write_all(b"RIFF")?;
    f.write_all(&(36_u32 + data_size).to_le_bytes())?;
    f.write_all(b"WAVE")?;

    // ── fmt  chunk (16 bytes, PCM = 1) ───────────────────────────────────────
    f.write_all(b"fmt ")?;
    f.write_all(&16_u32.to_le_bytes())?;
    f.write_all(&1_u16.to_le_bytes())?;            // AudioFormat: PCM
    f.write_all(&CHANNELS.to_le_bytes())?;
    f.write_all(&SAMPLE_RATE.to_le_bytes())?;
    f.write_all(&byte_rate.to_le_bytes())?;
    f.write_all(&block_align.to_le_bytes())?;
    f.write_all(&bits.to_le_bytes())?;

    // ── data chunk ───────────────────────────────────────────────────────────
    f.write_all(b"data")?;
    f.write_all(&data_size.to_le_bytes())?;
    f.write_all(pcm_data)?;

    let duration = pcm_data.len() as f64
        / (SAMPLE_RATE as f64 * CHANNELS as f64 * SAMPLE_WIDTH as f64);

    println!(
        "[PI] Saved: {} ({:.2}s, {} bytes)",
        filename.display(),
        duration,
        pcm_data.len()
    );

    Ok(())
}

// ── Serial I/O primitives ─────────────────────────────────────────────────────

/// Read exactly one byte.  Returns `Ok(Some(b))` on success, `Ok(None)` on
/// timeout (maps to Python's `ser.read(1)` returning `b""`).
fn read_byte(port: &mut Box<dyn SerialPort>) -> io::Result<Option<u8>> {
    let mut buf = [0u8; 1];
    match port.read(&mut buf) {
        Ok(n) if n > 0 => Ok(Some(buf[0])),
        Ok(_)           => Ok(None),
        Err(e) if e.kind() == io::ErrorKind::TimedOut => Ok(None),
        Err(e)          => Err(e),
    }
}

/// Read exactly `n` bytes, looping over partial reads the way Python's
/// `ser.read(n)` does.  Returns `Ok(None)` on timeout / short read.
fn read_exact_n(port: &mut Box<dyn SerialPort>, n: usize) -> io::Result<Option<Vec<u8>>> {
    let mut buf   = vec![0u8; n];
    let mut total = 0usize;

    while total < n {
        match port.read(&mut buf[total..]) {
            Ok(0)  => return Ok(None),
            Ok(k)  => total += k,
            Err(e) if e.kind() == io::ErrorKind::TimedOut => return Ok(None),
            Err(e) => return Err(e),
        }
    }

    Ok(Some(buf))
}

// ── Frame protocol ────────────────────────────────────────────────────────────

/// Scan the serial stream for the two-byte sync header `[0xAA, 0x55]`.
/// Returns `true` when found, `false` on timeout – mirrors `find_sync()`.
fn find_sync(port: &mut Box<dyn SerialPort>) -> io::Result<bool> {
    let mut state: u8 = 0;

    loop {
        match read_byte(port)? {
            None    => return Ok(false), // timeout
            Some(b) => match state {
                0 => {
                    if b == SYNC_BYTE_1 { state = 1; }
                }
                1 => {
                    if b == SYNC_BYTE_2 {
                        return Ok(true);
                    } else if b == SYNC_BYTE_1 {
                        state = 1; // could be start of new sync
                    } else {
                        state = 0;
                    }
                }
                _ => state = 0,
            },
        }
    }
}

/// Read one complete frame.
///
/// | Return value       | Python equivalent      | Meaning               |
/// |--------------------|------------------------|-----------------------|
/// | `Ok(None)`         | `None`                 | timeout               |
/// | `Ok(Some([]))`     | `b''`                  | end-of-stream marker  |
/// | `Ok(Some(data))`   | `bytes`                | valid PCM frame       |
fn read_frame(port: &mut Box<dyn SerialPort>) -> io::Result<Option<Vec<u8>>> {
    if !find_sync(port)? {
        return Ok(None);
    }

    // 2-byte big-endian length
    let len_bytes = match read_exact_n(port, 2)? {
        Some(b) => b,
        None    => return Ok(None),
    };

    let frame_len = ((len_bytes[0] as usize) << 8) | len_bytes[1] as usize;

    if frame_len == 0 {
        return Ok(Some(vec![])); // end-of-stream
    }

    // Read PCM payload
    match read_exact_n(port, frame_len)? {
        Some(data) => Ok(Some(data)),
        None       => Ok(None),
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let dir = output_dir();
    ensure_output_dir(&dir).expect("[PI] Failed to create output directory");

    println!("[PI] Audio Receiver Starting");
    println!("[PI] Serial: {} @ {} baud", SERIAL_PORT_PATH, BAUD_RATE);
    println!("[PI] Output: {}", dir.display());
    println!("[PI] Waiting for audio from ESP32...");
    println!();

    // Open the UART
    let mut port = serialport::new(SERIAL_PORT_PATH, BAUD_RATE)
        .data_bits(serialport::DataBits::Eight)
        .parity(serialport::Parity::None)
        .stop_bits(serialport::StopBits::One)
        .timeout(Duration::from_millis(RECEIVE_TIMEOUT_MS))
        .open()
        .expect("[PI] Failed to open serial port");

    // Flush any stale bytes in the driver buffer
    port.clear(serialport::ClearBuffer::Input).ok();

    // ── Signal handling (SIGINT / SIGTERM) ────────────────────────────────────
    // `ctrlc` registers handlers for both SIGINT and SIGTERM by default when
    // the "termination" feature is enabled – identical to the Python version's
    // `signal.signal(SIGINT/SIGTERM, handler)`.
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        println!("\n[PI] Shutting down...");
        r.store(false, Ordering::SeqCst);
    })
    .expect("[PI] Error setting signal handler");

    // ── Main receive loop ─────────────────────────────────────────────────────
    let mut audio_buf: Vec<u8> = Vec::new();
    let mut session_active = false;

    while running.load(Ordering::SeqCst) {
        let frame = match read_frame(&mut port) {
            Ok(f)  => f,
            Err(e) => {
                eprintln!("[PI] Serial error: {}", e);
                break;
            }
        };

        match frame {
            // ── Timeout ──────────────────────────────────────────────────────
            None => {
                if session_active && !audio_buf.is_empty() {
                    let path = generate_filename(&dir);
                    if let Err(e) = save_wav(&path, &audio_buf) {
                        eprintln!("[PI] Error saving WAV: {}", e);
                    }
                    audio_buf.clear();
                    session_active = false;
                    println!("[PI] Session ended (timeout). Waiting for next...");
                }
            }

            // ── End-of-stream (LEN == 0) ──────────────────────────────────
            Some(data) if data.is_empty() => {
                if !audio_buf.is_empty() {
                    let path = generate_filename(&dir);
                    if let Err(e) = save_wav(&path, &audio_buf) {
                        eprintln!("[PI] Error saving WAV: {}", e);
                    }
                    audio_buf.clear();
                }
                session_active = false;
                println!("[PI] ESP32 going to sleep. Waiting for next wake-up...");
            }

            // ── Valid PCM frame ───────────────────────────────────────────
            Some(data) => {
                if !session_active {
                    session_active = true;
                    let ts = utc_timestamp();
                    println!(
                        "[PI] New audio session started at {}",
                        time_from_timestamp(&ts)
                    );
                }

                audio_buf.extend_from_slice(&data);

                let duration = audio_buf.len() as f64
                    / (SAMPLE_RATE as f64 * CHANNELS as f64 * SAMPLE_WIDTH as f64);

                print!(
                    "\r[PI] Recording: {:.1}s ({} bytes)   ",
                    duration,
                    audio_buf.len()
                );
                io::stdout().flush().ok();
            }
        }
    }

    // ── Flush remaining audio on clean shutdown ───────────────────────────────
    if !audio_buf.is_empty() {
        let path = generate_filename(&dir);
        if let Err(e) = save_wav(&path, &audio_buf) {
            eprintln!("[PI] Error saving WAV: {}", e);
        }
    }

    println!("[PI] Receiver stopped.");
}
