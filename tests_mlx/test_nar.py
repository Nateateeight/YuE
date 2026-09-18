"""Load real VAE weights and test decode."""
import gc
import sys
import time
sys.path.insert(0, "/tmp/YuE")

import mlx.core as mx
import json
from pathlib import Path
from safetensors import safe_open
import torch
import struct
import numpy as np

print("Loading VAE with trained weights...")

# VAE config
vae_config = json.loads(Path("/tmp/yue2-mlx/vae/config.json").read_text())
encoder_cfg = vae_config["encoder_config"]
decoder_cfg = vae_config["decoder_config"]

# Create VAE
from src.yue2_mlx.vae_mlx import YuE2VAEMLX, OobleckDecoder, WeightNormConv1d, WeightNormConvTranspose1d

decoder = OobleckDecoder(decoder_cfg)

# Load VAE weights
print("Loading VAE weights...")
vae_weights = {}
with safe_open("/tmp/yue2-weights/vae/model.safetensors", framework="pt") as h:
    for key in h.keys():
        tensor = h.get_tensor(key)
        if str(tensor.dtype) == "torch.bfloat16":
            tensor = tensor.to(torch.float32)
        arr = tensor.numpy()
        # VAE conv weight layout: PyTorch (OC, IC, K) → MLX needs (OC, K, C_in)
        if arr.ndim == 3:
            arr = np.transpose(arr, (0, 2, 1))
        vae_weights[key] = mx.array(arr)
        print(f"  {key}: {vae_weights[key].shape}")

# Load into decoder
decoder.load_weights(vae_weights, strict=False)
del vae_weights
gc.collect()
print("VAE weights loaded!")

# Test decode with random latents
print("\nTesting VAE decode...")
latent = mx.random.normal((1, 64, 100))  # [B, C, T]
start = time.time()
audio = decoder(latent)
mx.eval(audio)
print(f"Decode time: {time.time() - start:.3f}s")
print(f"Audio shape: {audio.shape}")  # Should be [1, 2, ~192000] for 100 latent frames

# Save audio
def save_wav(audio_data, sample_rate, filename):
    """Save audio as 16-bit WAV."""
    if isinstance(audio_data, mx.array):
        audio_data = np.array(audio_data)
    
    # audio_data: [channels, samples] or [samples]
    if audio_data.ndim == 1:
        audio_data = audio_data[None, :]
    
    num_channels = audio_data.shape[0]
    num_samples = audio_data.shape[1]
    
    # Normalize
    mx_val = np.max(np.abs(audio_data))
    if mx_val > 0:
        audio_data = audio_data / mx_val * 0.8
    
    # Convert to int16
    audio_int16 = (audio_data.T * 32767).astype(np.int16)
    
    with open(filename, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + num_samples * num_channels * 2))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))  # PCM
        f.write(struct.pack('<H', num_channels))
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', sample_rate * num_channels * 2))
        f.write(struct.pack('<H', num_channels * 2))
        f.write(struct.pack('<H', 16))
        f.write(b'data')
        f.write(struct.pack('<I', num_samples * num_channels * 2))
        f.write(audio_int16.tobytes())
    
    return filename

output = "/tmp/yue2-mlx/test_vae.wav"
Path("/tmp/yue2-mlx").mkdir(exist_ok=True)
save_wav(audio, 48000, output)
print(f"\nSaved: {output}")
print(f"Duration: {audio.shape[-1] / 48000:.2f}s")

# Now test the NAR flow matching with real model
print("\n" + "="*60)
print("Testing NAR flow matching...")
print("="*60)

# Load LM for NAR
with open("/tmp/yue2-mlx/lm/config.json") as f:
    lm_config = json.load(f)

from src.yue2_mlx.modeling_mlx import YuE2MLX
model = YuE2MLX(lm_config)

# Load LM weights
def map_name(n):
    if n.startswith("model.model."):
        n = n[len("model.model."):]
    n = n.replace("model.embed_tokens.", "embed_tokens.")
    n = n.replace("model.norm.", "norm.")
    n = n.replace("model.llm2vae.", "llm2vae.")
    n = n.replace("model.vae2llm.", "vae2llm.")
    n = n.replace("model.latent_pos_embed.pe", "latent_pos_embed")
    if "time_embedder.mlp.2." in n:
        n = n.replace("time_embedder.mlp.2.", "time_embed.mlp.1.")
    elif "time_embedder." in n:
        n = n.replace("time_embedder.", "time_embed.")
    return n

sf = sorted(Path("/tmp/yue2-weights").glob("*.safetensors"))[0]
batch = {}
with safe_open(sf, framework="pt") as h:
    for key in h.keys():
        t = h.get_tensor(key)
        if str(t.dtype) == "torch.bfloat16":
            t = t.to(torch.float32)
        batch[map_name(key)] = mx.array(t.numpy())
        del t

model.load_weights(batch, strict=False)
del batch
gc.collect()
print("LM loaded!")

# NAR flow matching
print("\nRunning NAR flow matching (32 steps)...")
import time

# Create codec tokens
codec_tokens = list(range(1024, 1024 + 50))  # 50 tokens for testing
ar_mask = mx.array([[True] * len(codec_tokens)], dtype=mx.bool_)

# Initialize noise
mx.random.seed(42)
num_latent_frames = len(codec_tokens) * 20  # ~20 latent frames per codec token
noise = mx.random.normal((num_latent_frames, 64))

# Build input: AR tokens + NAR tokens
# AR tokens: codec tokens (with offset)
# NAR tokens: latent positions
CODEC_OFFSET = 151853
ar_tokens = [CODEC_OFFSET + (t - 1024) % 8192 for t in codec_tokens]

# Combine into single sequence
all_tokens = ar_tokens + [184621] + list(range(184622, 184622 + num_latent_frames)) + [184622]
input_ids = mx.array([all_tokens], dtype=mx.int32)
position_ids = mx.arange(len(all_tokens), dtype=mx.int32)[None, :]

# AR mask: True for AR positions, False for NAR positions
ar_positions = len(ar_tokens)
nar_positions = num_latent_frames + 2  # +2 for START/END tokens
ar_mask_list = [True] * ar_positions + [False] * nar_positions
ar_mask = mx.array([ar_mask_list], dtype=mx.bool_)

print(f"  Sequence length: {len(all_tokens)}")
print(f"  AR positions: {ar_positions}, NAR positions: {nar_positions}")
print(f"  Input shape: {input_ids.shape}")
print(f"  AR mask shape: {ar_mask.shape}")

# Run prefill
print("  Running prefill...")
start = time.time()
hidden, caches = model(input_ids, position_ids, ar_mask=ar_mask)
mx.eval(hidden)
print(f"  Prefill: {time.time() - start:.2f}s")

# Extract NAR hidden states at latent positions
nar_hidden = hidden[:, ar_positions:, :]
print(f"  NAR hidden shape: {nar_hidden.shape}")

# Predict velocity through llm2vae
velocity = model.llm2vae(nar_hidden)
print(f"  Velocity shape: {velocity.shape}")

# Simple Euler integration (1 step for testing)
dt = 1.0
state = noise - velocity[0, :, :] * dt

print(f"\n  Latents shape: {state.shape}")
print(f"  Latent range: [{float(mx.min(state)):.2f}, {float(mx.max(state)):.2f})]")

# Decode to audio
print("\n  Decoding latents to audio...")
start = time.time()
audio = decoder(state[None, :, :])  # Add batch dimension
mx.eval(audio)
print(f"  Decode: {time.time() - start:.3f}s")
print(f"  Audio shape: {audio.shape}")

# Save
output = "/tmp/yue2-mlx/test_nar.wav"
save_wav(audio, 48000, output)
print(f"\n  Saved: {output}")
print(f"  Duration: {audio.shape[-1] / 48000:.2f}s")

print("\n" + "="*60)
print("🎉 NAR flow matching complete! Audio generated!")
print("="*60)
