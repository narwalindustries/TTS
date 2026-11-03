import torch
import torchaudio
import scipy.io.wavfile as wavfile
import numpy as np

# Load reference audio
ref_audio, ref_sr = torchaudio.load("reference.mp3")
if ref_sr != 16000:
    ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, 16000)
if ref_audio.shape[0] > 1:
    ref_audio = ref_audio.mean(dim=0, keepdim=True)

# 1. Compute reference speaker embedding (256-d)
ref_spec = torch.stft(ref_audio, n_fft=512, hop_length=128, win_length=512, return_complex=True)
ref_mag = torch.abs(ref_spec)
mean_energy = ref_mag.mean(dim=-1).squeeze(0)[:128]
std_energy = ref_mag.std(dim=-1).squeeze(0)[:128]
features = mean_energy + std_energy

torch.manual_seed(42)
proj = torch.randn(128, 256)
speaker_lock = torch.matmul(features, proj)
speaker_lock = speaker_lock / (torch.norm(speaker_lock, p=2) + 1e-8)

# Save speaker_lock.bin
import struct
with open("models/speaker_lock.bin", "wb") as f:
    f.write(struct.pack("<256f", *speaker_lock.tolist()))
print("[*] Saved models/speaker_lock.bin from reference.mp3 (256 floats)")

# Compute spectral envelope of reference voice
ref_env = mean_energy / (mean_energy.sum() + 1e-8)
print(f"[*] Reference audio profile computed from {ref_audio.shape[1]/16000:.2f}s of audio")

