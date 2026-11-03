import os
import torch
import glob
import numpy as np
import scipy.io.wavfile as wav
from transformers import VitsModel, AutoTokenizer
from openvoice_cli.api import ToneColorConverter

os.makedirs("outputs", exist_ok=True)

# 1. Generate base Hindi speech using MMS-TTS
print("Loading MMS Hindi model...")
mms_hin = VitsModel.from_pretrained("facebook/mms-tts-hin")
tok_hin = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")

hindi_text = "नमस्ते! मैं आपकी दी हुई आवाज़ में हिंदी बोल रही हूँ। अब यह आवाज़ बिल्कुल साफ़ और महिला जैसी सुनाई देगी।"
inputs = tok_hin(hindi_text, return_tensors="pt")
with torch.no_grad():
    mms_out = mms_hin(**inputs).waveform[0].cpu().numpy()

src_hindi_path = "outputs/hi_mms_base.wav"
wav.write(src_hindi_path, mms_hin.config.sampling_rate, (mms_out * 32767).astype(np.int16))
print(f"Base Hindi audio generated at {src_hindi_path}")

# 2. Load OpenVoice ToneColorConverter
ckpt_dir = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--myshell-ai--OpenVoiceV2/snapshots/*/converter'))[0]
config_path = os.path.join(ckpt_dir, 'config.json')
checkpoint_path = os.path.join(ckpt_dir, 'checkpoint.pth')

print("Loading OpenVoice v2 ToneColorConverter...")
tone_color_converter = ToneColorConverter(config_path, device='cpu')
tone_color_converter.load_ckpt(checkpoint_path)
tone_color_converter.watermark_model = None

ref_wav = "reference.wav"
out_wav = "outputs/hi_cloned_reference.wav"

print(f"Extracting target speaker embedding from {ref_wav}...")
tgt_se = tone_color_converter.extract_se(ref_wav)

print(f"Extracting source speaker embedding from {src_hindi_path}...")
src_se = tone_color_converter.extract_se(src_hindi_path)

print(f"Converting Hindi voice timbre to match female reference...")
converted_audio = tone_color_converter.convert(src_hindi_path, src_se, tgt_se, output_path=None, tau=0.3)

sr = tone_color_converter.hps.data.sampling_rate
norm_audio = converted_audio / (np.max(np.abs(converted_audio)) + 1e-7)
wav.write(out_wav, sr, (norm_audio * 32767).astype(np.int16))
print(f"Conversion complete! Saved to {out_wav} (sr={sr}, duration={len(norm_audio)/sr:.2f}s)")

