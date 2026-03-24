import torch
import torch.nn as nn
import torchaudio
from resemblyzer import VoiceEncoder

print("Loading Resemblyzer...")
# Initialize the original encoder
original_encoder = VoiceEncoder(verbose=False)

class SmartVoiceAuth(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.encoder = original_model
        
        # Define the Mel Spectrogram transform exactly as Resemblyzer expects
        # 16kHz sample rate, 25ms window (400 samples), 10ms hop (160 samples)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=40,
            window_fn=torch.hamming_window,
            power=1.0  # Resemblyzer uses magnitude spectrograms, not power
        )

    def forward(self, waveform):
        # waveform shape: (Batch, Samples)
        
        # 1. Generate Mel Spectrogram
        # Result shape: (Batch, n_mels, time)
        mel = self.mel_transform(waveform)
        
        # 2. Log compression (Resemblyzer expects log-mel)
        # Add small epsilon to avoid log(0)
        mel = torch.log(mel + 1e-6)
        
        # 3. Transpose for LSTM: (Batch, time, n_mels)
        mel = mel.transpose(1, 2)
        
        # 4. Run Encoder
        # The encoder outputs normalized embeddings
        embed = self.encoder(mel)
        return embed

# Wrap the model
model = SmartVoiceAuth(original_encoder.model)
model.eval()

# Create dummy input (1 batch, 16000 samples = 1 second)
dummy_input = torch.randn(1, 16000)

print("Exporting to ONNX...")
torch.onnx.export(
    model,
    dummy_input,
    "voice_auth.onnx",
    input_names=["audio"],
    output_names=["embedding"],
    dynamic_axes={
        "audio": {1: "time"}, # Variable length audio
        "embedding": {0: "batch"}
    },
    opset_version=17
)
print("Success! 'voice_auth.onnx' created.")
