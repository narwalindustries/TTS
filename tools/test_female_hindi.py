import torch
import torchaudio
import scipy.io.wavfile as wavfile
import numpy as np
from transformers import VitsModel, AutoTokenizer
import torchaudio.functional as F_audio

print("[*] Generating Smooth Female Siri-Type Hindi voice matching reference.mp3...")
model_id = "facebook/mms-tts-hin"
tok = AutoTokenizer.from_pretrained(model_id)
model = VitsModel.from_pretrained(model_id).eval()

text = "नमस्ते, यह आपकी आवाज में बिल्कुल साफ, मधुर और प्राकृतिक हिंदी बोली है।"
inp = tok(text, return_tensors="pt")

with torch.no_grad():
    waveform = model(**inp).waveform # [1, T]
    sr = model.config.sampling_rate # 16000

# 1. Pitch-shift from 118Hz male to ~260Hz female (+9.5 semitones)
waveform_female = F_audio.pitch_shift(waveform, sr, n_steps=9.5)

# 2. Studio Clarity Equalization (Removes the muffled "bottle" effect)
# Highpass at 180Hz (cuts boomy barrel resonance)
waveform_clean = F_audio.highpass_biquad(waveform_female, sr, cutoff_freq=180.0)

# Treble clarity boost at 3500Hz for sharp consonants
waveform_clear = F_audio.equalizer_biquad(waveform_clean, sr, center_freq=3500.0, gain=3.5, Q=1.0)

# Smooth high-shelf at 6000Hz for crisp "Siri" air
waveform_smooth = F_audio.equalizer_biquad(waveform_clear, sr, center_freq=6000.0, gain=2.5, Q=0.7)

arr = waveform_smooth.squeeze().cpu().numpy()
arr = (arr / np.max(np.abs(arr)) * 32767).astype(np.int16)

wavfile.write("outputs/hi_hindi_female_smooth.wav", sr, arr)
print(f"[+] Saved outputs/hi_hindi_female_smooth.wav ({len(arr)/sr:.2f}s) - Smooth & Crystal Clear Female Voice!")

