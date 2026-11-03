import os
import sys
import glob
import numpy as np
import scipy.io.wavfile as wav
import torch
import librosa

sys.stdout.reconfigure(encoding='utf-8')
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

from openvoice_cli.api import ToneColorConverter
from kokoro import KPipeline
from transformers import VitsModel, AutoTokenizer

# 1. Initialize ToneColorConverter
ckpt_dir = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--myshell-ai--OpenVoiceV2/snapshots/*/converter'))[0]
config_path = os.path.join(ckpt_dir, 'config.json')
checkpoint_path = os.path.join(ckpt_dir, 'checkpoint.pth')

print("Loading OpenVoice v2 ToneColorConverter...")
converter = ToneColorConverter(config_path, device='cpu')
converter.load_ckpt(checkpoint_path)
converter.watermark_model = None

ref_wav = "reference.wav"
print(f"Extracting target speaker embedding from {ref_wav}...")
tgt_se = converter.extract_se(ref_wav)
target_f0 = 300.0  # Airebela / Siri female median F0

# Save speaker_lock.bin
se_np = tgt_se.squeeze().cpu().numpy()
se_np.tofile("models/speaker_lock.bin")
print(f"Locked speaker identity saved to models/speaker_lock.bin ({se_np.nbytes} bytes)")

def convert_and_save(src_wav_path, out_wav_path, adapt_pitch=True):
    sr_conv = converter.hps.data.sampling_rate
    audio_raw, sr_in = librosa.load(src_wav_path, sr=sr_conv)
    
    if adapt_pitch:
        f0_vals, _, _ = librosa.pyin(audio_raw, fmin=80, fmax=500)
        curr_f0 = float(np.nanmedian(f0_vals))
        if curr_f0 > 50 and curr_f0 < 220:
            semitones = 12 * np.log2(target_f0 / curr_f0)
            print(f"  [Pitch Adapt] Base F0: {curr_f0:.1f} Hz -> Shifting +{semitones:.1f} semitones to target {target_f0:.1f} Hz")
            audio_raw = librosa.effects.pitch_shift(audio_raw, sr=sr_conv, n_steps=semitones)
            wav.write(src_wav_path, sr_conv, (audio_raw * 32767).astype(np.int16))
    
    src_se = converter.extract_se(src_wav_path)
    converted = converter.convert(src_wav_path, src_se, tgt_se, output_path=None, tau=0.3)
    norm = converted / (np.max(np.abs(converted)) + 1e-7)
    wav.write(out_wav_path, sr_conv, (norm * 32767).astype(np.int16))
    
    f0_res, _, _ = librosa.pyin(norm, fmin=80, fmax=500)
    final_f0 = float(np.nanmedian(f0_res))
    duration = len(norm) / sr_conv
    print(f"  [Done] {out_wav_path} -> Duration: {duration:.2f}s | F0: {final_f0:.1f} Hz (Female Target: {target_f0:.1f} Hz)\n")

# A. English (via Kokoro 24k HD base -> ToneColor)
print("=== Synthesizing English (Studio Female HD) ===")
pipe = KPipeline(lang_code='a')
en_text = "Mmm, here is the secret recipe for the most decadent chocolate cake ever. I am speaking completely in your reference female voice."
en_chunks = []
for _, _, a in pipe(en_text, voice='af_heart', speed=1.0):
    en_chunks.extend(a)
src_en = "outputs/en_temp.wav"
wav.write(src_en, 24000, np.array(en_chunks, dtype=np.float32))
convert_and_save(src_en, "outputs/en_female_reference_siri.wav", adapt_pitch=False)

# B. Hindi (via MMS Hindi -> Pitch Harmonization -> ToneColor)
print("=== Synthesizing Hindi (MMS Native -> Female Siri Harmonized) ===")
mms_hin = VitsModel.from_pretrained("facebook/mms-tts-hin")
tok_hin = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")
hi_text = "नमस्ते! मैं आपकी दी हुई आवाज़ में हिंदी बोल रही हूँ। अब यह आवाज़ बिल्कुल साफ़ और महिला जैसी सुनाई दे रही है।"
inputs_hi = tok_hin(hi_text, return_tensors="pt")
with torch.no_grad():
    hi_out = mms_hin(**inputs_hi).waveform[0].cpu().numpy()
src_hi = "outputs/hi_temp.wav"
wav.write(src_hi, mms_hin.config.sampling_rate, (hi_out * 32767).astype(np.int16))
convert_and_save(src_hi, "outputs/hi_female_reference_siri.wav", adapt_pitch=True)

# C. Spanish
print("=== Synthesizing Spanish (MMS Spanish -> Female Siri Harmonized) ===")
mms_spa = VitsModel.from_pretrained("facebook/mms-tts-spa")
tok_spa = AutoTokenizer.from_pretrained("facebook/mms-tts-spa")
es_text = "¡Hola! Estoy hablando en español con tu voz de referencia femenina de forma muy clara y suave."
inputs_es = tok_spa(es_text, return_tensors="pt")
with torch.no_grad():
    es_out = mms_spa(**inputs_es).waveform[0].cpu().numpy()
src_es = "outputs/es_temp.wav"
wav.write(src_es, mms_spa.config.sampling_rate, (es_out * 32767).astype(np.int16))
convert_and_save(src_es, "outputs/es_female_reference_siri.wav", adapt_pitch=True)

# Clean up temp files
for tmp in [src_en, src_hi, src_es]:
    if os.path.exists(tmp):
        try: os.remove(tmp)
        except: pass

print("All requested voices successfully generated!")

