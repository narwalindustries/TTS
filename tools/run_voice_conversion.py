import os
import torch
import glob
import numpy as np
import scipy.io.wavfile as wav
from openvoice_cli.api import ToneColorConverter

ckpt_dir = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--myshell-ai--OpenVoiceV2/snapshots/*/converter'))[0]
config_path = os.path.join(ckpt_dir, 'config.json')
checkpoint_path = os.path.join(ckpt_dir, 'checkpoint.pth')

print("Loading ToneColorConverter...")
tone_color_converter = ToneColorConverter(config_path, device='cpu')
tone_color_converter.load_ckpt(checkpoint_path)
tone_color_converter.watermark_model = None

ref_wav = "reference.wav"
src_wav = "outputs/en_kokoro.wav"
out_wav = "outputs/en_cloned_reference.wav"

print(f"Extracting target speaker embedding from {ref_wav}...")
tgt_se = tone_color_converter.extract_se(ref_wav)

print(f"Extracting source speaker embedding from {src_wav}...")
src_se = tone_color_converter.extract_se(src_wav)

print(f"Converting voice timbre to match {ref_wav}...")
converted_audio = tone_color_converter.convert(src_wav, src_se, tgt_se, output_path=None, tau=0.3)

sr = tone_color_converter.hps.data.sampling_rate
# Normalize and save with scipy
norm_audio = converted_audio / (np.max(np.abs(converted_audio)) + 1e-7)
wav.write(out_wav, sr, (norm_audio * 32767).astype(np.int16))
print(f"Conversion complete! Saved to {out_wav} (sr={sr}, duration={len(norm_audio)/sr:.2f}s)")
