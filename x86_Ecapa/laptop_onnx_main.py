import onnxruntime as ort
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write, read
import os
import glob
import torch
import torchaudio
import time
import psutil

# The problematic line that caused the error has been REMOVED.

# --- Constants ---
VOICEPRINT_DB = "voiceprint_database_onnx"
FS = 16000
DURATION = 4

class FeatureExtractor:
    """
    Handles audio loading with Scipy and feature extraction with torchaudio.
    This is the robust, independent implementation.
    """
    def __init__(self, sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80):
        print("[-] Initializing manual feature extractor (using scipy + torchaudio)...")
        self.melspectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=n_mels)
        print("[-] Feature extractor ready.")
        
    def extract_features(self, audio_path):
        """Loads audio with Scipy and computes Log-Mel Spectrogram features."""
        fs, signal_numpy = read(audio_path)
        if signal_numpy.ndim > 1:
            signal_numpy = np.mean(signal_numpy, axis=1)
        signal = torch.from_numpy(signal_numpy).float().unsqueeze(0)
        if fs != FS:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=FS)
            signal = resampler(signal)
        mel_spec = self.melspectrogram(signal)
        log_mel_spec = torch.log(mel_spec + 1e-6)
        features = log_mel_spec.permute(0, 2, 1)
        return features.numpy()

class ONNXAuthenticator:
    """Handles the ONNX model inference."""
    def __init__(self, model_path):
        print("[-] Initializing ONNX Runtime session...")
        self.session = ort.InferenceSession(model_path)
        print("[-] ONNX Authenticator ready.")

    def extract_embedding(self, features):
        inputs = {'input_features': features}
        embedding = self.session.run(None, inputs)[0]
        return embedding

    def compare_embeddings(self, emb1, emb2):
        """Computes cosine similarity between two flattened embeddings."""
        emb1_flat = emb1.flatten()
        emb2_flat = emb2.flatten()
        dot_product = np.dot(emb1_flat, emb2_flat)
        norm_product = np.linalg.norm(emb1_flat) * np.linalg.norm(emb2_flat)
        return dot_product / norm_product

# --- Application Functions ---
def record_audio(filename):
    print(f"\n...Recording for {DURATION} seconds...")
    recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='float32')
    sd.wait()
    print("...Recording complete.")
    write(filename, FS, recording)

def enroll_speaker(feature_extractor, authenticator):
    username = input("Enter a username for enrollment: ").lower().strip()
    if not username: print("[!] Username cannot be empty."); return
    embedding_path = os.path.join(VOICEPRINT_DB, f"{username}.npy")
    if os.path.exists(embedding_path): print(f"[!] User '{username}' already exists."); return
    
    enroll_file = "temp_enroll.wav"
    record_audio(enroll_file)
    
    print("[-] Extracting features from audio...")
    features = feature_extractor.extract_features(enroll_file)
    print("[-] Creating voiceprint with ONNX model...")
    embedding = authenticator.extract_embedding(features)
    
    np.save(embedding_path, embedding)
    os.remove(enroll_file)
    print(f"[+] Enrollment successful for '{username}'.")

def identify_speaker_realtime(feature_extractor, authenticator, threshold=0.70):
    if not glob.glob(os.path.join(VOICEPRINT_DB, "*.npy")): print("\n[!] No speakers enrolled."); return

    process = psutil.Process(os.getpid())
    
    verify_file = "temp_verify.wav"
    record_audio(verify_file)
    
    process.cpu_percent(interval=None)

    start_time = time.perf_counter()
    live_features = feature_extractor.extract_features(verify_file)
    feature_latency_ms = (time.perf_counter() - start_time) * 1000
    
    start_time = time.perf_counter()
    live_embedding = authenticator.extract_embedding(live_features)
    inference_latency_ms = (time.perf_counter() - start_time) * 1000
    
    cpu_usage = process.cpu_percent(interval=None)
    ram_usage_mb = process.memory_info().rss / (1024 * 1024)

    os.remove(verify_file)
    
    best_score, best_match = -1.0, None
    for embedding_path in glob.glob(os.path.join(VOICEPRINT_DB, "*.npy")):
        enrolled_embedding = np.load(embedding_path)
        score = authenticator.compare_embeddings(enrolled_embedding, live_embedding).item()
        username = os.path.basename(embedding_path).replace(".npy", "")
        print(f"  - Compared with {username}: Score = {score:.4f}")
        if score > best_score:
            best_score, best_match = score, username

    print(f"\n[-] Best Similarity Score: {best_score:.4f} (with {best_match})")
    if best_score > threshold:
        print(f">>> RESULT: MATCH - Welcome, {best_match}!")
    else:
        print(f">>> RESULT: NO MATCH - Unknown Speaker")

    print("\n--- Performance Metrics for this Run ---")
    print(f"  Feature Extraction Latency: {feature_latency_ms:.2f} ms")
    print(f"  ONNX Inference Latency:     {inference_latency_ms:.2f} ms")
    print(f"  End-to-End Latency:         {(feature_latency_ms + inference_latency_ms):.2f} ms")
    print(f"  CPU Usage during processing: {cpu_usage:.2f} %")
    print(f"  Current RAM Footprint:      {ram_usage_mb:.2f} MB")
    print("----------------------------------------")

# --- Main Execution Block ---
if __name__ == "__main__":
    onnx_model_path = 'ecapa_tdnn.onnx'
    if not os.path.exists(onnx_model_path):
        print(f"[ERROR] Missing '{onnx_model_path}'. Please run the conversion script first.")
        exit()

    os.makedirs(VOICEPRINT_DB, exist_ok=True)
    
    model_size_mb = os.path.getsize(onnx_model_path) / (1024 * 1024)
    print("--- System Initialized ---")
    print(f"  Model on Disk: {model_size_mb:.2f} MB")
    print("--------------------------")
    
    feature_extractor = FeatureExtractor()
    authenticator = ONNXAuthenticator(onnx_model_path)

    while True:
        print("\n--- Independent ONNX Voice Biometric System (with Benchmarking) ---")
        print("1. Enroll New Speaker")
        print("2. Identify Speaker")
        print("3. Exit")
        choice = input("Enter choice: ")
        if choice == '1': enroll_speaker(feature_extractor, authenticator)
        elif choice == '2': identify_speaker_realtime(feature_extractor, authenticator)
        elif choice == '3': break
        else: print("[!] Invalid choice.")