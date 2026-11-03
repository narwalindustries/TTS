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

# Load locked voice tensor for English, Japanese, Chinese, Spanish, French, Italian, Portuguese
locked_tensor = torch.load("models/locked_voice.pt", weights_only=True)
print("Loaded locked style tensor from models/locked_voice.pt")

def synthesize_with_natural_pauses(pipe, text, voice, speed=0.92, sample_rate=24000):
    """
    Synthesizes speech with human breathing pauses at clause and sentence boundaries.
    Prevents rushed, breathless, continuous monotone speech.
    """
    # Delimiters: commas, semicolons, colons, periods, exclamations, questions, and Hindi danda
    parts = re.split(r'([,;:\.!?।]+)', text)
    chunks = []
    i = 0
    while i < len(parts):
        seg = parts[i].strip()
        punct = parts[i+1].strip() if i+1 < len(parts) else ''
        i += 2
        phrase = f"{seg}{punct}".strip()
        if not phrase:
            continue
            
        # Synthesize clause/phrase
        for _, _, audio in pipe(phrase, voice=voice, speed=speed):
            chunks.append(np.array(audio, dtype=np.float32))
            
        # Insert natural human pauses based on punctuation
        if punct in [',', ';', ':']:
            # Natural clause breathing pause: ~180 ms
            chunks.append(np.zeros(int(0.18 * sample_rate), dtype=np.float32))
        elif any(p in punct for p in ['.', '!', '?', '।']):
            # Natural sentence boundary pause: ~320 ms
            chunks.append(np.zeros(int(0.32 * sample_rate), dtype=np.float32))
            
    if not chunks:
        return np.zeros(sample_rate, dtype=np.float32)
        
    full_audio = np.concatenate(chunks)
    # Peak normalize to 0.95 to eliminate digital clipping
    full_audio = full_audio / (np.max(np.abs(full_audio)) + 1e-7) * 0.95
    return full_audio

# Pipelines cache
pipes = {}
def get_pipeline(lang_code):
    if lang_code not in pipes:
        print(f"Initializing KPipeline for lang_code='{lang_code}'...")
        pipes[lang_code] = KPipeline(lang_code=lang_code)
    return pipes[lang_code]

# --- 1. ENGLISH, JAPANESE, CHINESE (DO NOT ALTER: generate original approved style) ---
print("\n=== 1. Preserving Approved Core Languages (EN, JA, ZH) ===")
approved_languages = [
    {
        "lang_code": "a",
        "lang_name": "English (US)",
        "filename": "outputs/en_locked_female_af_heart.wav",
        "text": "Hello! This voice is now permanently locked across all languages. Notice the crisp pronunciation, warm tone, and controlled cadence.",
        "speed": 1.0
    },
    {
        "lang_code": "j",
        "lang_name": "Japanese",
        "filename": "outputs/ja_locked_female_af_heart.wav",
        "text": "こんにちは！この同じ声が日本語でも固定されました。とても自然で聞き取りやすい発音です。",
        "speed": 1.0
    },
    {
        "lang_code": "z",
        "lang_name": "Mandarin Chinese",
        "filename": "outputs/zh_locked_female_af_heart.wav",
        "text": "你好！这同一个声音现在已锁定为中文，语调自然，发音清晰而准确。",
        "speed": 1.0
    }
]

for item in approved_languages:
    pipe = get_pipeline(item["lang_code"])
    audio_chunks = []
    for _, _, a in pipe(item["text"], voice=locked_tensor, speed=item["speed"]):
        audio_chunks.extend(a)
    arr = np.array(audio_chunks, dtype=np.float32)
    arr = arr / (np.max(np.abs(arr)) + 1e-7) * 0.95
    wav.write(item["filename"], 24000, (arr * 32767).astype(np.int16))
    print(f"  [SAVED] {item['lang_name']}: {item['filename']} ({len(arr)/24000:.2f}s)")

# --- 2. MULTILINGUAL ENHANCEMENT: SPANISH, FRENCH, ITALIAN, PORTUGUESE (NATURAL PAUSES & CLEAR PACING) ---
print("\n=== 2. Enhanced Multilingual Voices with Natural Pauses & Pacing ===")
enhanced_languages = [
    {
        "lang_code": "e",
        "lang_name": "Spanish",
        "filename": "outputs/es_locked_female_af_heart.wav",
        # Natural punctuation breaks inserted to guarantee human-like prosodic breathing
        "text": "¡Hola! La misma voz femenina, ahora está fija para el español. Mantiene una entonación natural, clara, y perfectamente controlada."
    },
    {
        "lang_code": "f",
        "lang_name": "French",
        "filename": "outputs/fr_locked_female_af_heart.wav",
        "text": "Bonjour! Cette même voix est désormais configurée pour le français, avec une diction naturelle, fluide, et une clarté remarquable."
    },
    {
        "lang_code": "i",
        "lang_name": "Italian",
        "filename": "outputs/it_locked_female_af_heart.wav",
        "text": "Ciao! Questa stessa voce è ora perfettamente calibrata per l'italiano, con una dizione naturale, limpida, e un ritmo molto piacevole."
    },
    {
        "lang_code": "p",
        "lang_name": "Portuguese (BR)",
        "filename": "outputs/pt_locked_female_af_heart.wav",
        "text": "Olá! Esta mesma voz agora está calibrada para o português, mantendo a clareza, o ritmo suave, e pausas totalmente naturais."
    }
]

for item in enhanced_languages:
    pipe = get_pipeline(item["lang_code"])
    arr = synthesize_with_natural_pauses(pipe, item["text"], voice=locked_tensor, speed=0.92)
    wav.write(item["filename"], 24000, (arr * 32767).astype(np.int16))
    print(f"  [SAVED] {item['lang_name']}: {item['filename']} ({len(arr)/24000:.2f}s, Natural Pauses & Articulate Pacing)")

# --- 3. HINDI FIX: NATIVE INDIAN HINDI FEMALE VOICE & COLLOQUIAL PHONETICS ---
print("\n=== 3. Native Indian Hindi Speech Generation (No Foreign Accent, Natural Human Pauses) ===")
pipe_hi = get_pipeline('h')
voice_hf_alpha = pipe_hi.load_voice('hf_alpha')
voice_hf_beta = pipe_hi.load_voice('hf_beta')

hindi_samples = [
    {
        "filename": "outputs/hi_locked_female_native.wav",
        "voice": voice_hf_alpha,
        "title": "Hindi Primary (hf_alpha Native Indian Female)",
        "text": "नमस्ते! अब आपकी ये पसंदीदा आवाज़, हिंदी में भी पूरी तरह से तैयार हो चुकी है। इसका लहज़ा, बोलने की गति, और उच्चारण, बिल्कुल सटीक और साफ़ है।"
    },
    {
        "filename": "outputs/hi_native_female_conversational.wav",
        "voice": voice_hf_alpha,
        "title": "Hindi Natural Conversational (hf_alpha)",
        "text": "नमस्ते! अब मैं हिंदी में भी बात कर सकती हूँ। मेरा बोलने का अंदाज़, आवाज़ की मिठास, और हर शब्द का उच्चारण, बिल्कुल साफ़ और सहज है।"
    },
    {
        "filename": "outputs/hi_native_female_clear_beta.wav",
        "voice": voice_hf_beta,
        "title": "Hindi Alternate Warm Tone (hf_beta)",
        "text": "नमस्ते! अब आपकी ये पसंदीदा आवाज़, हिंदी में भी पूरी तरह से तैयार हो चुकी है। इसका लहज़ा, बोलने की गति, और उच्चारण, बिल्कुल सटीक और साफ़ है।"
    }
]

for item in hindi_samples:
    arr = synthesize_with_natural_pauses(pipe_hi, item["text"], voice=item["voice"], speed=0.90)
    wav.write(item["filename"], 24000, (arr * 32767).astype(np.int16))
    print(f"  [SAVED] {item['title']}: {item['filename']} ({len(arr)/24000:.2f}s)")

print("\n=== All Audio Generations Successfully Completed! ===")
