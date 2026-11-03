import os
import sys
import re
import numpy as np
import scipy.io.wavfile as wav
import torch
from kokoro import KPipeline

sys.stdout.reconfigure(encoding='utf-8')
os.makedirs("outputs", exist_ok=True)

pipe = KPipeline(lang_code='h')
voice_alpha = torch.load("models/locked_voice_hindi.pt", weights_only=True)

# Test 3 snappy, natural tempo settings:
# 1. Normal Natural (Speed 1.0, 120ms clause pause, 240ms sentence pause)
# 2. Crisp Conversational (Speed 1.04, 100ms clause pause, 200ms sentence pause)
# 3. Slightly relaxed (Speed 0.97, 140ms clause pause, 260ms sentence pause)

variations = [
    ("hi_tempo_natural_1_00.wav", 1.00, 0.12, 0.24, "Natural Fluid (Speed 1.00)"),
    ("hi_tempo_crisp_1_04.wav", 1.04, 0.10, 0.20, "Crisp Lively (Speed 1.04)"),
    ("hi_tempo_balanced_0_98.wav", 0.98, 0.13, 0.25, "Balanced Smooth (Speed 0.98)")
]

text = "नमस्ते! अब आपकी ये पसंदीदा आवाज़, हिंदी में पूरी तरह से तैयार है। इसका लहज़ा, बोलने की गति, और उच्चारण, बिल्कुल सटीक और साफ़ है।"

delimiters = r'([,;.!?!।]+)'
parts = re.split(delimiters, text)

for fname, spd, pause_short, pause_long, label in variations:
    chunks = []
    i = 0
    sample_rate = 24000
    while i < len(parts):
        seg = parts[i].strip()
        punct = parts[i+1].strip() if i+1 < len(parts) else ''
        i += 2
        phrase = f"{seg}{punct}".strip()
        if not phrase:
            continue
        for _, ph, a in pipe(phrase, voice=voice_alpha, speed=spd):
            chunks.append(np.array(a, dtype=np.float32))
        if punct in [',', ';']:
            chunks.append(np.zeros(int(pause_short * sample_rate), dtype=np.float32))
        elif any(p in punct for p in ['.', '!', '?', '।']):
            chunks.append(np.zeros(int(pause_long * sample_rate), dtype=np.float32))
            
    arr = np.concatenate(chunks)
    arr = arr / (np.max(np.abs(arr)) + 1e-7) * 0.95
    out_file = os.path.abspath(f"outputs/{fname}")
    wav.write(out_file, sample_rate, (arr * 32767).astype(np.int16))
    print(f"Generated {fname} ({label}): {len(arr)/sample_rate:.2f}s")

# Default outputs/hi_locked_female.wav to the Natural Fluid (1.00)
import shutil
shutil.copy("outputs/hi_tempo_natural_1_00.wav", "outputs/hi_locked_female.wav")
shutil.copy("outputs/hi_tempo_natural_1_00.wav", "outputs/hi_locked_female_hf_alpha.wav")
print("Updated outputs/hi_locked_female.wav with Natural Fluid tempo")

