import sys
import os
import re
import numpy as np
import scipy.io.wavfile as wav
import torch
from kokoro import KPipeline

sys.stdout.reconfigure(encoding='utf-8')
os.makedirs("outputs", exist_ok=True)

pipe = KPipeline(lang_code='h')
voice_heart = torch.load("models/locked_voice.pt", weights_only=True)
voice_alpha = pipe.load_voice('hf_alpha')

# Generate both orientations of 15 / 85:
# Option A: 85% Heart + 15% Alpha (Primary Heart voice, softened dental D & T)
# Option B: 15% Heart + 85% Alpha (Strong Native Hindi pronunciation, infused with Heart timbre)
candidates = [
    ("outputs/hi_heart_85_alpha_15.wav", 0.85, 0.15, "85% Heart + 15% Alpha"),
    ("outputs/hi_heart_15_alpha_85.wav", 0.15, 0.85, "15% Heart + 85% Alpha")
]

text = "नमस्ते! अब आपकी ये पसंदीदा आवाज़, हिंदी में पूरी तरह से तैयार है। इसका लहज़ा, बोलने की गति, और त तथा द का उच्चारण, बिल्कुल सटीक और साफ़ है।"

delimiters = r'([,;.!?!।]+)'
parts = re.split(delimiters, text)

for out_file, w_h, w_a, label in candidates:
    v = w_h * voice_heart + w_a * voice_alpha
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
    wav.write(out_file, 24000, (arr * 32767).astype(np.int16))
    print(f"Saved {out_file} ({label}, {len(arr)/24000:.2f}s)")

# Also copy 85% Heart + 15% Alpha into hi_locked_female_af_heart.wav as default
import shutil
shutil.copy("outputs/hi_heart_85_alpha_15.wav", "outputs/hi_locked_female_af_heart.wav")
print("Updated outputs/hi_locked_female_af_heart.wav with 85/15 blend")

