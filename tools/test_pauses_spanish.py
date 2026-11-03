import sys
import os
import re
import numpy as np
import scipy.io.wavfile as wav
import torch
from kokoro import KPipeline

sys.stdout.reconfigure(encoding='utf-8')
os.makedirs("outputs", exist_ok=True)

pipe = KPipeline(lang_code='e') # Spanish
voice = torch.load("models/locked_voice.pt", weights_only=True)
text = "¡Hola! La misma voz femenina ahora está fija para el español, manteniendo una entonación natural, clara y perfectamente controlada."

# Split on punctuation while capturing delimiter
delimiters = r'([,;.!?]+)'
parts = re.split(delimiters, text)

chunks = []
i = 0
while i < len(parts):
    seg = parts[i].strip()
    punct = parts[i+1].strip() if i+1 < len(parts) else ''
    i += 2
    phrase = f"{seg}{punct}".strip()
    if not phrase:
        continue
    print(f"Phrase: '{phrase}'")
    for _, _, a in pipe(phrase, voice=voice, speed=0.92):
        chunks.append(np.array(a, dtype=np.float32))
    
    # Human breathing pauses
    if punct in [',', ';']:
        # 160ms clause pause
        chunks.append(np.zeros(int(0.16 * 24000), dtype=np.float32))
    elif any(p in punct for p in ['.', '!', '?']):
        # 300ms sentence pause
        chunks.append(np.zeros(int(0.30 * 24000), dtype=np.float32))

arr = np.concatenate(chunks)
arr = arr / (np.max(np.abs(arr)) + 1e-7) * 0.95
wav.write("outputs/test_es_pauses.wav", 24000, (arr * 32767).astype(np.int16))
print(f"Saved outputs/test_es_pauses.wav ({len(arr)/24000:.2f}s)")

