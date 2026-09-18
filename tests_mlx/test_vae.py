"""Quick test for YuE2 VAE MLX port."""
import time
import sys
sys.path.insert(0, "/tmp/YuE")

import mlx.core as mx
from src.yue2_mlx.vae_mlx import YuE2VAEMLX

encoder_config = {
    "in_channels": 2,
    "channels": 64,
    "c_mults": [1, 2, 4, 8, 16, 32],
    "strides": [2, 2, 4, 4, 5, 6],
    "latent_dim": 128,
    "use_snake": True,
}

decoder_config = {
    "out_channels": 2,
    "channels": 64,
    "c_mults": [1, 2, 4, 8, 16, 32],
    "strides": [2, 2, 4, 4, 5, 6],
    "latent_dim": 64,
    "use_snake": True,
    "final_tanh": False,
}

print("Creating VAE...")
vae = YuE2VAEMLX(encoder_config, decoder_config)

def count_params(obj):
    if isinstance(obj, mx.array):
        return obj.size
    if isinstance(obj, dict):
        return sum(count_params(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(count_params(v) for v in obj)
    return 0

print(f"Decoder params: {count_params(vae.decoder.parameters()) / 1e6:.2f}M")

# Test decoder with random latent
print("Testing decode...")
latent = mx.random.normal((1, 64, 100))
start = time.time()
audio = vae.decode(latent)
mx.eval(audio)
elapsed = time.time() - start
print(f"Decode: {elapsed:.3f}s, shape: {audio.shape}")

# Test encoder
print("Testing encode...")
audio_in = mx.random.normal((1, 2, 192000))  # 4 seconds
start = time.time()
latent_out = vae.encode(audio_in)
mx.eval(latent_out)
elapsed = time.time() - start
print(f"Encode: {elapsed:.3f}s, shape: {latent_out.shape}")

print("\nALL VAE TESTS PASSED")
