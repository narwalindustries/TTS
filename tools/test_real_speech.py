import torch
import torchaudio
from transformers import VitsModel, AutoTokenizer

model = VitsModel.from_pretrained('facebook/mms-tts-hin')
tok = AutoTokenizer.from_pretrained('facebook/mms-tts-hin')

text = "नमस्ते, यह असली आवाज़ का परीक्षण है।"
inputs = tok(text, return_tensors='pt')

with torch.no_grad():
    output = model(**inputs).waveform

import os
import scipy.io.wavfile as wavfile
import numpy as np

os.makedirs('outputs', exist_ok=True)
arr = output.squeeze().cpu().numpy()
# Normalize to int16
arr = (arr / np.max(np.abs(arr)) * 32767).astype(np.int16)
wavfile.write('outputs/test_hindi_real.wav', model.config.sampling_rate, arr)
print(f"Generated real Hindi speech: {arr.shape} samples at {model.config.sampling_rate} Hz")
