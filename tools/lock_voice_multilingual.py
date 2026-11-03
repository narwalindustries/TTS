import os
import sys
import numpy as np
import scipy.io.wavfile as wav
import torch
from kokoro import KPipeline

sys.stdout.reconfigure(encoding='utf-8')
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 1. Permanently lock the voice profile to af_heart
LOCKED_VOICE = "af_heart"
print(f"=== Locking Voice Profile to '{LOCKED_VOICE}' (from outputs/en_english_female_native.wav) ===")

# Save the locked voice tensor to models/locked_voice.pt
pipe_temp = KPipeline(lang_code='a')
locked_tensor = pipe_temp.load_voice(LOCKED_VOICE)
torch.save(locked_tensor, "models/locked_voice.pt")
print(f"Saved locked voice style tensor to models/locked_voice.pt (shape: {locked_tensor.shape})")

# 2. Multilingual Sentences: strictly controlled accent, authentic native slang & prosody
languages = [
    {
        "lang_code": "a",
        "lang_name": "English (US)",
        "filename": "outputs/en_locked_female_af_heart.wav",
        "text": "Hello! This voice is now permanently locked across all languages. Notice the crisp pronunciation, warm tone, and controlled cadence."
    },
    {
        "lang_code": "h",
        "lang_name": "Hindi",
        "filename": "outputs/hi_locked_female_af_heart.wav",
        "text": "नमस्ते! अब आपकी यह पसंदीदा आवाज़ हिंदी में भी पूरी तरह से लॉक हो चुकी है। इसका लहज़ा, गति और उच्चारण बिल्कुल सटीक और साफ़ है।"
    },
    {
        "lang_code": "e",
        "lang_name": "Spanish",
        "filename": "outputs/es_locked_female_af_heart.wav",
        "text": "¡Hola! La misma voz femenina ahora está fija para el español, manteniendo una entonación natural, clara y perfectamente controlada."
    },
    {
        "lang_code": "f",
        "lang_name": "French",
        "filename": "outputs/fr_locked_female_af_heart.wav",
        "text": "Bonjour! Cette même voix est désormais verrouillée pour le français, avec un accent authentique, fluide et maîtrisé."
    },
    {
        "lang_code": "i",
        "lang_name": "Italian",
        "filename": "outputs/it_locked_female_af_heart.wav",
        "text": "Ciao! Questa stessa voce è ora bloccata per l'italiano, con una dizione naturale, chiara e molto piacevole."
    },
    {
        "lang_code": "p",
        "lang_name": "Portuguese (BR)",
        "filename": "outputs/pt_locked_female_af_heart.wav",
        "text": "Olá! Esta mesma voz agora está travada para o português, mantendo a clareza, o ritmo suave e a dicção perfeita."
    },
    {
        "lang_code": "j",
        "lang_name": "Japanese",
        "filename": "outputs/ja_locked_female_af_heart.wav",
        "text": "こんにちは！この同じ声が日本語でも固定されました。とても自然で聞き取りやすい発音です。"
    },
    {
        "lang_code": "z",
        "lang_name": "Mandarin Chinese",
        "filename": "outputs/zh_locked_female_af_heart.wav",
        "text": "你好！这同一个声音现在已锁定为中文，语调自然，发音清晰而准确。"
    }
]

pipelines = {'a': pipe_temp}

print("\n=== Generating Multilingual Speech with Locked Voice ===")

for item in languages:
    code = item["lang_code"]
    name = item["lang_name"]
    out_file = item["filename"]
    text = item["text"]
    
    print(f"\nProcessing {name} (lang_code='{code}')...")
    
    if code not in pipelines:
        pipelines[code] = KPipeline(lang_code=code)
    
    pipe = pipelines[code]
    audio_chunks = []
    
    try:
        # Strictly controlled speech rate: speed=1.0 for articulate cadence
        gen = pipe(text, voice=locked_tensor, speed=1.0)
        for _, _, audio in gen:
            audio_chunks.extend(audio)
            
        if len(audio_chunks) > 0:
            arr = np.array(audio_chunks, dtype=np.float32)
            # Normalize peak to 0.95 (prevent digital clipping)
            arr = arr / (np.max(np.abs(arr)) + 1e-7) * 0.95
            wav.write(out_file, 24000, (arr * 32767).astype(np.int16))
            dur = len(arr) / 24000.0
            print(f"  [SUCCESS] {out_file} ({dur:.2f}s, 24kHz HD)")
        else:
            print(f"  [FAIL] No audio produced for {name}")
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")

print("\n=== Voice Lock Generation Finished! ===")

