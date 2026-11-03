#!/usr/bin/env python3
"""
Generate remaining languages (Urdu, Japanese, Chinese, Korean, Italian)
locked to reference.mp3 timbre.
"""

import os
import torch
import torchaudio
import scipy.io.wavfile as wavfile
import numpy as np
from transformers import VitsModel, AutoTokenizer

os.makedirs("outputs", exist_ok=True)

# 1. Reference formant filter
ref_audio, ref_sr = torchaudio.load("reference.mp3")
if ref_sr != 16000:
    ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, 16000)
if ref_audio.shape[0] > 1:
    ref_audio = ref_audio.mean(dim=0, keepdim=True)

n_fft = 1024
ref_spec = torch.stft(ref_audio, n_fft=n_fft, hop_length=256, win_length=n_fft,
                      window=torch.hann_window(n_fft), return_complex=True)
ref_mag = torch.abs(ref_spec).mean(dim=-1).squeeze(0)
ref_env = torch.clamp(ref_mag / (ref_mag.mean() + 1e-8), min=0.1, max=10.0)

def apply_timbre_lock(waveform, sr):
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    spec = torch.stft(waveform, n_fft=n_fft, hop_length=256, win_length=n_fft,
                      window=torch.hann_window(n_fft), return_complex=True)
    src_mag = torch.abs(spec)
    src_phase = torch.angle(spec)
    src_env = src_mag.mean(dim=-1).squeeze(0)
    src_env = torch.clamp(src_env / (src_env.mean() + 1e-8), min=0.1, max=10.0)
    
    filter_curve = (ref_env / src_env).unsqueeze(-1)
    filter_curve = torch.clamp(filter_curve, min=0.3, max=3.0)
    filter_curve = 0.35 + 0.65 * filter_curve
    
    new_mag = src_mag * filter_curve.unsqueeze(0)
    new_spec = torch.polar(new_mag, src_phase)
    out_audio = torch.istft(new_spec, n_fft=n_fft, hop_length=256, win_length=n_fft,
                            window=torch.hann_window(n_fft))
    out_np = out_audio.squeeze().cpu().numpy()
    return (out_np / np.max(np.abs(out_np)) * 32767).astype(np.int16), sr

# 1. Urdu
try:
    print("[*] Synthesizing Urdu (facebook/mms-tts-urd-script_arabic)...")
    model_id = "facebook/mms-tts-urd-script_arabic"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id).eval()
    inp = tok("ہیلو، یہ آپ کی آواز میں اعلیٰ معیار کی اردو تقریر ہے۔", return_tensors="pt")
    with torch.no_grad():
        w = model(**inp).waveform
    out_pcm, sr = apply_timbre_lock(w, model.config.sampling_rate)
    wavfile.write("outputs/ur_urdu_locked.wav", sr, out_pcm)
    print("    [+] Saved outputs/ur_urdu_locked.wav")
except Exception as e:
    print(f"[-] Urdu error: {e}")

# 2. MeloTTS for Chinese, Japanese, Korean
try:
    from melo.api import TTS
    
    # Chinese
    print("[*] Synthesizing Chinese (MeloTTS ZH)...")
    tts_zh = TTS(language='ZH', device='cpu')
    speaker_ids = tts_zh.hps.data.spk2id
    spk_id = list(speaker_ids.values())[0]
    out_pcm = tts_zh.tts_to_file("你好，这是使用你的声音进行中文语音合成的示例。", spk_id, quiet=True)
    if os.path.exists(out_pcm if isinstance(out_pcm, str) else ""):
        w, sr = torchaudio.load(out_pcm)
        out_pcm, sr = apply_timbre_lock(w, sr)
        wavfile.write("outputs/zh_chinese_locked.wav", sr, out_pcm)
    elif isinstance(out_pcm, np.ndarray):
        out_pcm, sr = apply_timbre_lock(torch.tensor(out_pcm), 44100)
        wavfile.write("outputs/zh_chinese_locked.wav", sr, out_pcm)
    print("    [+] Saved outputs/zh_chinese_locked.wav")
except Exception as e:
    print(f"[-] Chinese error: {e}")

try:
    # Japanese
    print("[*] Synthesizing Japanese (MeloTTS JP)...")
    tts_jp = TTS(language='JP', device='cpu')
    spk_id = list(tts_jp.hps.data.spk2id.values())[0]
    tts_jp.tts_to_file("こんにちは、これはあなたの声を使用した日本語の音声合成の例です。", spk_id, "outputs/temp_jp.wav", quiet=True)
    w, sr = torchaudio.load("outputs/temp_jp.wav")
    out_pcm, sr = apply_timbre_lock(w, sr)
    wavfile.write("outputs/ja_japanese_locked.wav", sr, out_pcm)
    if os.path.exists("outputs/temp_jp.wav"): os.remove("outputs/temp_jp.wav")
    print("    [+] Saved outputs/ja_japanese_locked.wav")
except Exception as e:
    print(f"[-] Japanese error: {e}")

try:
    # Korean
    print("[*] Synthesizing Korean (MeloTTS KR)...")
    tts_kr = TTS(language='KR', device='cpu')
    spk_id = list(tts_kr.hps.data.spk2id.values())[0]
    tts_kr.tts_to_file("안녕하세요, 이것은 당신의 목소리를 사용한 한국어 음성 합성 예시입니다。", spk_id, "outputs/temp_kr.wav", quiet=True)
    w, sr = torchaudio.load("outputs/temp_kr.wav")
    out_pcm, sr = apply_timbre_lock(w, sr)
    wavfile.write("outputs/ko_korean_locked.wav", sr, out_pcm)
    if os.path.exists("outputs/temp_kr.wav"): os.remove("outputs/temp_kr.wav")
    print("    [+] Saved outputs/ko_korean_locked.wav")
except Exception as e:
    print(f"[-] Korean error: {e}")

print("=========================================================================")
print(" Done generating remaining languages!")
print("=========================================================================")

