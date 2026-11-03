import os
import torch
import glob
from openvoice_cli.api import ToneColorConverter

ckpt_dir = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--myshell-ai--OpenVoiceV2/snapshots/*/converter'))[0]
config_path = os.path.join(ckpt_dir, 'config.json')
checkpoint_path = os.path.join(ckpt_dir, 'checkpoint.pth')

print(f"Loading ToneColorConverter from {ckpt_dir}...")
tone_color_converter = ToneColorConverter(config_path, device='cpu')
tone_color_converter.watermark_model = None
tone_color_converter.load_ckpt(checkpoint_path)
print("ToneColorConverter successfully loaded!")

# Extract target speaker embedding from reference.wav
ref_wav = "reference.wav"
print(f"Extracting target speaker embedding from {ref_wav}...")
target_se = tone_color_converter.extract_se(ref_wav)
print(f"Target SE extracted! Shape: {target_se.shape}, norm: {torch.norm(target_se).item():.4f}")

# Save target SE as speaker_lock.bin
os.makedirs("models", exist_ok=True)
se_np = target_se.squeeze().cpu().numpy()
se_np.tofile("models/speaker_lock.bin")
print(f"Saved models/speaker_lock.bin ({se_np.nbytes} bytes)")
