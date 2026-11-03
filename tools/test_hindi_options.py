import os
import torch
import glob
import numpy as np
import scipy.io.wavfile as wav
import librosa
from transformers import VitsModel, AutoTokenizer
from openvoice_cli.api import ToneColorConverter

os.makedirs("outputs", exist_ok=True)

# 1. Generate MMS Hindi base speech
print("1. Generating base MMS Hindi audio...")
mms_hin = VitsModel.from_pretrained("facebook/mms-tts-hin")
tok_hin = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")

text = "नमस्ते! मैं आपकी दी हुई आवाज़ में हिंदी बोल रही हूँ। क्या मेरी आवाज़ बिल्कुल साफ़ और मधुर सुनाई दे रही है?"
inputs = tok_hin(text, return_tensors="pt")
with torch.no_grad():
    mms_out = mms_hin(**inputs).waveform[0].cpu().numpy()

sr_mms = mms_hin.config.sampling_rate
src_hindi_raw = "outputs/hi_mms_raw.wav"
wav.write(src_hindi_raw, sr_mms, (mms_out * 32767).astype(np.int16))
print(f"Base Hindi audio saved to {src_hindi_raw} ({len(mms_out)/sr_mms:.2f}s)")

# Measure base pitch
f0_base, _, _ = librosa.pyin(mms_out.astype(np.float32), sr=sr_mms, fmin=80, fmax=500)
base_f0_val = float(np.nanmedian(f0_base))
print(f"Base MMS Hindi F0 median: {base_f0_val:.1f} Hz")

# Target pitch is ~300 Hz from reference.wav
semitones = 12 * np.log2(290.0 / max(base_f0_val, 100))
print(f"Shifting base Hindi pitch by +{semitones:.1f} semitones to reach ~290 Hz...")
mms_pitched = librosa.effects.pitch_shift(mms_out.astype(np.float32), sr=sr_mms, n_steps=semitones)
src_hindi_pitched = "outputs/hi_mms_pitched.wav"
wav.write(src_hindi_pitched, sr_mms, (mms_pitched * 32767).astype(np.int16))

# 2. Load OpenVoice ToneColorConverter
print("2. Loading OpenVoice v2 ToneColorConverter...")
ckpt_dir = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--myshell-ai--OpenVoiceV2/snapshots/*/converter'))[0]
config_path = os.path.join(ckpt_dir, 'config.json')
checkpoint_path = os.path.join(ckpt_dir, 'checkpoint.pth')

tone_color_converter = ToneColorConverter(config_path, device='cpu')
tone_color_converter.load_ckpt(checkpoint_path)
tone_color_converter.watermark_model = None

ref_wav = "reference.wav"
print("Extracting target speaker embedding from reference.wav...")
tgt_se = tone_color_converter.extract_se(ref_wav)

# Option A: Convert raw MMS Hindi
print("Converting Option A (Direct MMS -> ToneColor)...")
src_se_raw = tone_color_converter.extract_se(src_hindi_raw)
out_a = tone_color_converter.convert(src_hindi_raw, src_se_raw, tgt_se, output_path=None, tau=0.3)
sr_conv = tone_color_converter.hps.data.sampling_rate
norm_a = out_a / (np.max(np.abs(out_a)) + 1e-7)
wav.write("outputs/hi_option_a_direct.wav", sr_conv, (norm_a * 32767).astype(np.int16))

# Option B: Convert pitch-shifted MMS Hindi
print("Converting Option B (Pitch-shifted -> ToneColor)...")
src_se_pitched = tone_color_converter.extract_se(src_hindi_pitched)
out_b = tone_color_converter.convert(src_hindi_pitched, src_se_pitched, tgt_se, output_path=None, tau=0.3)
norm_b = out_b / (np.max(np.abs(out_b)) + 1e-7)
wav.write("outputs/hi_female_siri_hindi.wav", sr_conv, (norm_b * 32767).astype(np.int16))

print("Both options generated successfully!")

