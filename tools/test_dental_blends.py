import sys
import os
import re
import numpy as np
import scipy.io.wavfile as wav
import torch
from kokoro import KPipeline

sys.stdout.reconfigure(encoding='utf-8')
os.makedirs("outputs/td_tests", exist_ok=True)

pipe = KPipeline(lang_code='h')
voice_heart = torch.load("models/locked_voice.pt", weights_only=True)
voice_alpha = pipe.load_voice('hf_alpha')

# Focused sentence with lots of T and D sounds:
# 'नमस्ते' (t), 'अब आपकी' (p, k), 'ये पसंदीदा' (p, s, n, d, d), 'तैयार' (t), 'गति' (t), 'सटीक और साफ़' (t, k, s, f)
text = "नमस्ते! अब आपकी ये पसंदीदा आवाज़, हिंदी में पूरी तरह से तैयार है। इसका लहज़ा, गति, और त तथा द का उच्चारण, बिल्कुल सटीक और साफ़ है।"

blends = [
    ("pure_heart_100", 1.0, 0.0),
    ("blend_heart_85_alpha_15", 0.85, 0.15),
    ("blend_heart_70_alpha_30", 0.70, 0.30),
    ("blend_heart_50_alpha_50", 0.50, 0.50),
    ("blend_heart_30_alpha_70", 0.30, 0.70),
    ("pure_alpha_100", 0.0, 1.0)
]

delimiters = r'([,;.!?!।]+)'
parts = re.split(delimiters, text)

for name, w_heart, w_alpha in blends:
    v = w_heart * voice_heart + w_alpha * voice_alpha
    chunks = []
    i = 0
    while i < len(parts):
        seg = parts[i].strip()
        punct = parts[i+1].strip() if i+1 < len(parts) else ''
        i += 2
        phrase = f"{seg}{punct}".strip()
        if not phrase:
            continue
        for _, _, a in pipe(phrase, voice=v, speed=0.91):
            chunks.append(np.array(a, dtype=np.float32))
        if punct in [',', ';']:
            chunks.append(np.zeros(int(0.18 * 24000), dtype=np.float32))
        elif any(p in punct for p in ['.', '!', '?', '।']):
            chunks.append(np.zeros(int(0.32 * 24000), dtype=np.float32))
            
    arr = np.concatenate(chunks)
    arr = arr / (np.max(np.abs(arr)) + 1e-7) * 0.95
    out_path = f"outputs/td_tests/{name}.wav"
    wav.write(out_path, 24000, (arr * 32767).astype(np.int16))
    print(f"Saved {out_path} ({len(arr)/24000:.2f}s)")

