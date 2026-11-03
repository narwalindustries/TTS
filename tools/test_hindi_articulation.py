import sys
import os
import re
import numpy as np
import scipy.io.wavfile as wav
import torch
from kokoro import KPipeline

sys.stdout.reconfigure(encoding='utf-8')
os.makedirs("outputs", exist_ok=True)

pipe = KPipeline(lang_code='h') # Hindi pipeline

test_cases = [
    {
        "id": "hf_alpha_colloquial_natural",
        "voice": "hf_alpha",
        "text": "नमस्ते! अब आपकी ये पसंदीदा आवाज़, हिंदी में भी पूरी तरह से तैयार हो चुकी है। इसका लहज़ा, बोलने की गति, और उच्चारण, बिल्कुल सटीक और साफ़ है।"
    },
    {
        "id": "hf_beta_colloquial_natural",
        "voice": "hf_beta",
        "text": "नमस्ते! अब आपकी ये पसंदीदा आवाज़, हिंदी में भी पूरी तरह से तैयार हो चुकी है। इसका लहज़ा, बोलने की गति, और उच्चारण, बिल्कुल सटीक और साफ़ है।"
    },
    {
        "id": "hf_alpha_formal_pure",
        "voice": "hf_alpha",
        "text": "नमस्कार! आपकी यह आवाज़, अब हिंदी भाषा में भी पूरी तरह से उपलब्ध है। हर एक शब्द का उच्चारण, गति, और अंदाज़, बिल्कुल स्पष्ट और स्वाभाविक है।"
    },
    {
        "id": "hf_alpha_everyday_conversational",
        "voice": "hf_alpha",
        "text": "नमस्ते! अब मैं हिंदी में भी बात कर सकती हूँ। मेरा बोलने का अंदाज़, आवाज़ की मिठास, और हर शब्द का उच्चारण, बिल्कुल साफ़ और सहज है।"
    }
]

delimiters = r'([,;.!?!।]+)'

for tc in test_cases:
    vid = tc["id"]
    vname = tc["voice"]
    raw_text = tc["text"]
    voice = pipe.load_voice(vname)
    
    parts = re.split(delimiters, raw_text)
    chunks = []
    i = 0
    print(f"\n--- Testing {vid} ---")
    while i < len(parts):
        seg = parts[i].strip()
        punct = parts[i+1].strip() if i+1 < len(parts) else ''
        i += 2
        phrase = f"{seg}{punct}".strip()
        if not phrase:
            continue
        print(f"Phrase: '{phrase}'")
        for g, ph, a in pipe(phrase, voice=voice, speed=0.90):
            print(f"  Phonemes: {ph}")
            chunks.append(np.array(a, dtype=np.float32))
        
        # Natural human breathing pauses
        if punct in [',', ';']:
            chunks.append(np.zeros(int(0.18 * 24000), dtype=np.float32))
        elif any(p in punct for p in ['.', '!', '?', '।']):
            chunks.append(np.zeros(int(0.32 * 24000), dtype=np.float32))
            
    arr = np.concatenate(chunks)
    arr = arr / (np.max(np.abs(arr)) + 1e-7) * 0.95
    out_file = f"outputs/{vid}.wav"
    wav.write(out_file, 24000, (arr * 32767).astype(np.int16))
    print(f"Saved {out_file} ({len(arr)/24000:.2f}s)")

