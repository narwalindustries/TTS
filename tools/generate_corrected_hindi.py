import os
import sys
import numpy as np
import scipy.io.wavfile as wav
import torch
from kokoro import KPipeline

sys.stdout.reconfigure(encoding='utf-8')
os.makedirs("outputs", exist_ok=True)

# Load locked voice tensor
locked_tensor = torch.load("models/locked_voice.pt", weights_only=True)
print("Loaded locked voice style tensor from models/locked_voice.pt")

p_hi = KPipeline(lang_code='h')

def preprocess_hindi(text: str) -> str:
    """
    Automatic Hindi Pronunciation & Prosody Enhancer.
    Replaces bookish orthography with natural colloquial pronunciations,
    fixes common loanwords, and adds natural rhythmic breathing pauses.
    """
    replacements = [
        # Conversational pronouns
        (r'\bयह\b', 'ये'),
        (r'\bवह\b', 'वो'),
        (r'\bइन्हें\b', 'इन्हे'),
        (r'\bउन्हें\b', 'उन्हे'),
        # Common phonetic smoothing
        (r'\bपसंदीदा\b', 'मनपसंद'),
        (r'\bलॉक\b', 'सेट'),
        (r'\bलहज़ा\b', 'अंदाज़'),
        # Breathing pauses before conjunctions
        (r' और ', ', और '),
        (r' लेकिन ', ', लेकिन '),
        (r' कि ', ', कि '),
        (r'  +', ' ')
    ]
    import re
    res = text
    for pat, rep in replacements:
        res = re.sub(pat, rep, res)
    return res.strip()

samples = [
    {
        "filename": "outputs/hi_corrected_natural.wav",
        "title": "Natural Conversational Hindi",
        "raw_text": "नमस्ते! अब आपकी ये आवाज़ हिंदी में भी पूरी तरह से सेट हो चुकी है। इसका उच्चारण, बोलने का ढंग और गति बिल्कुल साफ़ और स्वाभाविक है।"
    },
    {
        "filename": "outputs/hi_corrected_formal.wav",
        "title": "Crisp & Clear Articulate Hindi",
        "raw_text": "नमस्कार! आपकी आवाज़ अब हिंदी में भी तैयार है। इसमें हर शब्द का उच्चारण एकदम स्पष्ट, शुद्ध और सुनने में मधुर है।"
    },
    {
        "filename": "outputs/hi_corrected_conversational.wav",
        "title": "Friendly Everyday Hindi",
        "raw_text": "नमस्ते! मैं वही आवाज़ हूँ जो आपको पसंद आई थी। अब मैं हिंदी में भी बिल्कुल साफ़-साफ़ और बिना किसी रुकावट के बोल सकती हूँ।"
    },
    {
        "filename": "outputs/hi_corrected_original_fixed.wav",
        "title": "Original Sentence (Phonetically Corrected)",
        # Here we keep the exact same original thought but fix the phonetics (ये instead of यह, सेट instead of लॉक)
        "raw_text": "नमस्ते! अब आपकी ये पसंदीदा आवाज़ हिंदी में भी तैयार हो चुकी है। इसका बोलने का ढंग, गति और उच्चारण बिल्कुल सही और साफ़ है।"
    }
]

print("\n=== Generating Corrected Hindi Voice Outputs (Locked af_heart) ===")

for s in samples:
    fname = s["filename"]
    raw = s["raw_text"]
    processed = preprocess_hindi(raw)
    print(f"\nGenerating: {s['title']}")
    print(f"  Input:     {raw}")
    print(f"  Enhanced:  {processed}")
    
    audio_chunks = []
    for graphemes, phonemes, audio in p_hi(processed, voice=locked_tensor, speed=1.0):
        print(f"  Phonemes: {phonemes}")
        audio_chunks.extend(audio)
        
    arr = np.array(audio_chunks, dtype=np.float32)
    # Peak normalization to 0.95
    arr = arr / (np.max(np.abs(arr)) + 1e-7) * 0.95
    wav.write(fname, 24000, (arr * 32767).astype(np.int16))
    
    dur = len(arr) / 24000.0
    print(f"  [SUCCESS] Saved {fname} ({dur:.2f}s, 24kHz HD)")

print("\n=== All Corrected Hindi Outputs Generated! ===")

