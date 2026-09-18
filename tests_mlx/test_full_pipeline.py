"""Load real VAE weights and run NAR flow matching to generate audio."""
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

print("=" * 60)
print("YuE2 MLX - Full Pipeline Test")
print("=" * 60)

# ===== LOAD VAE =====
print("\n[1] Loading VAE with trained weights...")
from src.yue2_mlx.vae_mlx import OobleckDecoder, WeightNormConv1d

vae_config = json.loads(Path("/tmp/yue2-mlx/vae/config.json").read_text())
decoder = OobleckDecoder(vae_config["decoder_config"])

# Load VAE weights
vae_weights = {}
with safe_open("/tmp/yue2-weights/vae/model.safetensors", framework="pt") as h:
    for key in h.keys():
        t = h.get_tensor(key)
        if str(t.dtype) == "torch.bfloat16":
            t = t.to(torch.float32)
        arr = t.numpy()
        # VAE weight_norm: PyTorch has weight_g (1,1,1) and weight_v (OC, K, C_in)
        # MLX needs: weight = (weight_v / norm) * weight_g, where norm = ||weight_v||
        if ".weight_g" in key:
            continue  # Handle weight_g + weight_v together
        if ".weight_v" in key:
            # Get corresponding weight_g
            g_key = key.replace("weight_v", "weight_g")
            if g_key in h.keys():
                weight_g = h.get_tensor(g_key).numpy()  # (OC, 1, 1)
                weight_v = arr  # (OC, K, C_in)
                # Absorb weight_norm: weight = weight_g * weight_v / ||weight_v||
                # For fan-out norm: norm = sqrt(sum(weight_v^2))
                norm = np.sqrt(np.sum(weight_v**2, axis=(1, 2), keepdims=True))
                arr = (weight_v / norm) * weight_g
                key = key.replace("weight_v", "weight")
        elif "weight" in key and arr.ndim == 3:
            # Other conv weights: transpose layout
            arr = np.transpose(arr, (0, 2, 1))
        vae_weights[key] = mx.array(arr)

decoder.load_weights(vae_weights, strict=False)
del vae_weights
gc.collect()
print("  VAE loaded!")

# ===== LOAD LM =====
print("\n[2] Loading LM (3B)...")
with open("/tmp/yue2-mlx/lm/config.json") as f:
    config = json.load(f)

from src.yue2_mlx.modeling_mlx import YuE2MLX
model = YuE2MLX(config)

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

batch = {}
with safe_open("/tmp/yue2-weights/model.safetensors", framework="pt") as h:
    for key in h.keys():
        t = h.get_tensor(key)
        if str(t.dtype) == "torch.bfloat16":
            t = t.to(torch.float32)
        batch[map_name(key)] = mx.array(t.numpy())
        del t

model.load_weights(batch, strict=False)
del batch
gc.collect()
print("  LM loaded!")

# ===== NAR FLOW MATCHING =====
print("\n[3] Running NAR flow matching (32 steps)...")

# Create codec tokens
codec_tokens = list(range(1024, 1024 + 50))
CODEC_OFFSET = 151853
ar_tokens = [CODEC_OFFSET + (t - 1024) % 8192 for t in codec_tokens]
input_ids = mx.array([ar_tokens], dtype=mx.int32)
position_ids = mx.arange(len(ar_tokens), dtype=mx.int32)[None, :]

# Initialize noise
mx.random.seed(42)
num_latent_frames = 100
noise = mx.random.normal((num_latent_frames, 64))

# AR mask
ar_mask = mx.ones((1, len(ar_tokens)), dtype=mx.bool_)

# ODE solver
def solve_ode(model, input_ids, position_ids, initial_state, ar_mask, steps=32):
    """32-step midpoint ODE solver."""
    state = initial_state
    dt = 1.0 / steps
    
    for step in range(steps):
        t = 1.0 - step * dt
        # Logit-space conversion
        if t <= 0:
            raw_t = -20.0
        elif t >= 1:
            raw_t = 20.0
        else:
            raw_t = np.log(t / (1 - t))
        
        # First velocity
        v1 = model.forward_nar(input_ids, position_ids, state, raw_t, ar_mask)
        mx.eval(v1)
        
        # Midpoint
        mid = state - v1 * (dt / 2)
        
        # Midpoint velocity
        mid_t = 1.0 - (step + 0.5) * dt
        if mid_t <= 0:
            raw_mid = -20.0
        elif mid_t >= 1:
            raw_mid = 20.0
        else:
            raw_mid = np.log(mid_t / (1 - mid_t))
        
        v2 = model.forward_nar(input_ids, position_ids, mid, raw_mid, ar_mask)
        mx.eval(v2)
        
        # Full step
        state = state - v2 * dt
        
        if (step + 1) % 10 == 0:
            print(f"    Step {step+1}/{steps}")
    
    return state

start = time.time()
latents = solve_ode(model, input_ids, position_ids, noise, ar_mask, steps=32)
mx.eval(latents)
print(f"  ODE solve: {time.time() - start:.2f}s")
print(f"  Latents shape: {latents.shape}")
print(f"  Range: [{float(mx.min(latents)):.2f}, {float(mx.max(latents)):.2f}]")

# ===== VAE DECODE =====
print("\n[4] Decoding to audio...")
start = time.time()
# Latents are [T, 64], decoder expects [B, C, T] = [1, 64, T]
latents_for_decoder = latents.T[None, :, :]  # [1, 64, T]
audio = decoder(latents_for_decoder)
mx.eval(audio)
print(f"  Decode: {time.time() - start:.3f}s")
print(f"  Audio shape: {audio.shape}")

# Save audio
def save_wav(audio_data, sample_rate, filename):
    if isinstance(audio_data, mx.array):
        audio_data = np.array(audio_data)
    if audio_data.ndim == 1:
        audio_data = audio_data[None, :]
    mx_val = np.max(np.abs(audio_data))
    if mx_val > 0:
        audio_data = audio_data / mx_val * 0.8
    audio_int16 = (audio_data.T * 32767).astype(np.int16)
    num_channels = audio_data.shape[0]
    num_samples = audio_data.shape[1]
    with open(filename, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + num_samples * num_channels * 2))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<H', num_channels))
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', sample_rate * num_channels * 2))
        f.write(struct.pack('<H', num_channels * 2))
        f.write(struct.pack('<H', 16))
        f.write(b'data')
        f.write(struct.pack('<I', num_samples * num_channels * 2))
        f.write(audio_int16.tobytes())

Path("/tmp/yue2-mlx").mkdir(exist_ok=True)
output = "/tmp/yue2-mlx/yueml_test.wav"
save_wav(audio, 48000, output)
print(f"\n{'='*60}")
print(f"AUDIO GENERATED: {output}")
print(f"{'='*60}")
print(f"Duration: {audio.shape[-1] / 48000:.2f}s")
print(f"Sample rate: 48000Hz")
print(f"Channels: {audio.shape[0]}")
print(f"\nNote: This is a simplified test with random codec tokens.")
print(f"Real songs need proper text conditioning and CoT planning.")
