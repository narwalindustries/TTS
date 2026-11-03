import torch
import torchaudio
import scipy.io.wavfile as wavfile
import numpy as np
from transformers import VitsModel, AutoTokenizer
import torchaudio.functional as F_audio

model_id = "facebook/mms-tts-hin"
tok = AutoTokenizer.from_pretrained(model_id)
model = VitsModel.from_pretrained(model_id).eval()

text = "नमस्ते, यह आपकी आवाज में बिल्कुल साफ, मधुर और प्राकृतिक हिंदी बोली है।"
inp = tok(text, return_tensors="pt")

with torch.no_grad():
    waveform = model(**inp).waveform
    sr = model.config.sampling_rate

# Natural warm female Siri pitch (+5.5 semitones: 120Hz -> ~200Hz)
waveform_female = F_audio.pitch_shift(waveform, sr, n_steps=5.5)

# Studio clarity cleanup: remove bottle mud, boost vocal clarity
waveform_clean = F_audio.highpass_biquad(waveform_female, sr, cutoff_freq=160.0)
waveform_clear = F_audio.equalizer_biquad(waveform_clean, sr, center_freq=3200.0, gain=3.0, Q=1.0)
waveform_smooth = F_audio.equalizer_biquad(waveform_clear, sr, center_freq=7000.0, gain=2.0, Q=0.7)

arr = waveform_smooth.squeeze().cpu().numpy()
arr = (arr / np.max(np.abs(arr)) * 32767).astype(np.int16)

wavfile.write("outputs/hi_hindi_female_siri.wav", sr, arr)
print(f"[+] Saved outputs/hi_hindi_female_siri.wav ({len(arr)/sr:.2f}s) at natural Siri female pitch!")

