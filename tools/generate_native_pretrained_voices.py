import os
import sys
import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf
import librosa
from kokoro import KPipeline

sys.stdout.reconfigure(encoding='utf-8')
os.makedirs("outputs", exist_ok=True)

test_cases = [
    {
        "lang_code": "h",
        "voice": "hf_alpha",
        "lang_name": "Hindi",
        "filename": "outputs/hi_hindi_female_native.wav",
        "text": "नमस्ते! मैं आपके सिस्टम की डिफ़ॉल्ट प्रीट्रेंड महिला आवाज़ हूँ। यह आवाज़ बहुत साफ़, सहज और प्राकृतिक है।"
    },
    {
        "lang_code": "h",
        "voice": "hf_beta",
        "lang_name": "Hindi (Beta Voice)",
        "filename": "outputs/hi_hindi_female_beta.wav",
        "text": "नमस्ते! यह दूसरी प्रीट्रेंड महिला आवाज़ है। आप दोनों में से जो सबसे अच्छी लगे उसे चुन सकते हैं।"
    },
    {
        "lang_code": "a",
        "voice": "af_heart",
        "lang_name": "English (Studio HD)",
        "filename": "outputs/en_english_female_native.wav",
        "text": "Hello! I am the default studio-quality pretrained female voice. Notice how clean, smooth, and natural the pronunciation is, with zero robotic artifacts."
    },
    {
        "lang_code": "a",
        "voice": "af_bella",
        "lang_name": "English (Siri Style)",
        "filename": "outputs/en_english_female_bella.wav",
        "text": "Hey there! This is the Bella female voice profile. It is crisp, friendly, and articulate, perfect for conversational assistants."
    },
    {
        "lang_code": "e",
        "voice": "ef_dora",
        "lang_name": "Spanish",
        "filename": "outputs/es_spanish_female_native.wav",
        "text": "¡Hola! Esta es la voz femenina predeterminada en español, con una articulación clara y natural."
    },
    {
        "lang_code": "f",
        "voice": "ff_siwis",
        "lang_name": "French",
        "filename": "outputs/fr_french_female_native.wav",
        "text": "Bonjour! Ceci est la voix féminine native par défaut, douce et parfaitement intelligible."
    }
]

pipelines = {}

print("=== Generating Default Pretrained Native Female Voices ===")

for tc in test_cases:
    code = tc["lang_code"]
    voice = tc["voice"]
    print(f"\nProcessing {tc['lang_name']} ({voice})...")
    
    if code not in pipelines:
        print(f"  Initializing pipeline for lang_code='{code}'...")
        pipelines[code] = KPipeline(lang_code=code)
    
    pipeline = pipelines[code]
    audio_chunks = []
    
    try:
        generator = pipeline(tc["text"], voice=voice, speed=1.0)
        for _, _, audio in generator:
            audio_chunks.extend(audio)
        
        if len(audio_chunks) > 0:
            audio_arr = np.array(audio_chunks, dtype=np.float32)
            # Normalize to 0.95 peak
            peak = np.max(np.abs(audio_arr)) + 1e-7
            audio_arr = audio_arr / peak * 0.95
            
            wav.write(tc["filename"], 24000, (audio_arr * 32767).astype(np.int16))
            
            dur = len(audio_arr) / 24000.0
            print(f"  SUCCESS! Saved {tc['filename']} ({dur:.2f}s, 24kHz HD)")
        else:
            print(f"  WARNING: No audio chunks produced for {tc['lang_name']}")
    except Exception as e:
        print(f"  ERROR generating {tc['lang_name']}: {e}")

print("\n=== Generation Finished! ===")

