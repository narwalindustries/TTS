import os
import soundfile as sf
from kokoro import KPipeline

print("[*] Initializing Kokoro Female Siri-type voice...")
pipeline = KPipeline(lang_code='a')

text = "Hello! This is a crystal clear, smooth female voice, exactly like Siri. Every word is completely clear and natural."
generator = pipeline(text, voice='af_heart', speed=1.0, split_pattern=r'\n+')

for i, (gs, ps, audio) in enumerate(generator):
    sf.write('outputs/en_female_siri_crystal.wav', audio, 24000)
    print(f"    [+] Saved outputs/en_female_siri_crystal.wav ({len(audio)/24000:.2f}s)")
    break

