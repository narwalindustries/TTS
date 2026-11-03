import os
import sys
import re
import numpy as np
import scipy.io.wavfile as wav
import torch
from kokoro import KPipeline

sys.stdout.reconfigure(encoding='utf-8')
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 1. Initialize Hindi Pipeline and extract pure hf_alpha
print("=== Initializing Hindi Pipeline with Pure hf_alpha ===")
pipe = KPipeline(lang_code='h')
voice_alpha = pipe.load_voice('hf_alpha')

# Save locked voice for Hindi
torch.save(voice_alpha, "models/locked_voice_hindi.pt")
print("Saved models/locked_voice_hindi.pt (shape:", voice_alpha.shape, ")")

# 2. Text with authentic Hindi phonetics and conversational rhythm
text = "नमस्ते! अब आपकी ये पसंदीदा आवाज़, हिंदी में पूरी तरह से तैयार है। इसका लहज़ा, बोलने की गति, और उच्चारण, बिल्कुल सटीक और साफ़ है।"

delimiters = r'([,;.!?!।]+)'
parts = re.split(delimiters, text)

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
    print(f"Synthesizing phrase: '{phrase}'")
    for _, ph, a in pipe(phrase, voice=voice_alpha, speed=0.91):
        print(f"  Phonemes: {ph}")
        chunks.append(np.array(a, dtype=np.float32))
    
    # Natural human breathing pauses
    if punct in [',', ';']:
        chunks.append(np.zeros(int(0.18 * sample_rate), dtype=np.float32))
    elif any(p in punct for p in ['.', '!', '?', '।']):
        chunks.append(np.zeros(int(0.32 * sample_rate), dtype=np.float32))

arr = np.concatenate(chunks)
arr = arr / (np.max(np.abs(arr)) + 1e-7) * 0.95

# Save primary Hindi file
primary_file = os.path.abspath("outputs/hi_locked_female_hf_alpha.wav")
wav.write(primary_file, sample_rate, (arr * 32767).astype(np.int16))
print(f"Saved {primary_file} ({len(arr)/sample_rate:.2f}s, 24kHz HD)")

# Also mirror to outputs/hi_locked_female.wav
alias_file = os.path.abspath("outputs/hi_locked_female.wav")
wav.write(alias_file, sample_rate, (arr * 32767).astype(np.int16))
print(f"Saved {alias_file} ({len(arr)/sample_rate:.2f}s, 24kHz HD)")

# Clean up any residual test/blend files
for f in ["outputs/hi_heart_15_alpha_85.wav", "outputs/hi_heart_85_alpha_15.wav", "outputs/hi_locked_female_af_heart.wav"]:
    if os.path.exists(f):
        try:
            os.remove(f)
            print(f"Cleaned up {f}")
        except Exception:
            pass

print("=== Hindi Pure Alpha Configuration Completed! ===")

