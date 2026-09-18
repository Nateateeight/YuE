"""Quick test: create a tiny YuE2MLX and run a forward pass."""
import time
import math
import mlx.core as mx
import json
import sys
sys.path.insert(0, "/tmp/YuE")

from src.yue2_mlx.modeling_mlx import YuE2MLX

# Tiny config for testing
config = {
    "hidden_size": 256,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "intermediate_size": 512,
    "vocab_size": 1000,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "max_position_embeddings": 512,
    "latent_dim": 64,
    "max_latent_frames": 512,
    "timestep_shift": 1.0,
    "latent_type": "vae",
    "tie_word_embeddings": False,
}

print("Creating model...")
model = YuE2MLX(config)



# Test forward pass
print("Testing forward pass...")
input_ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
position_ids = mx.arange(5, dtype=mx.int32)[None, :]

start = time.time()
logits, caches = model(input_ids, position_ids)
mx.eval(logits)
elapsed = time.time() - start
print(f"Forward pass: {elapsed:.3f}s")
print(f"Logits shape: {logits.shape}")
print(f"Num caches: {len(caches)}")

# Test single-token decode
print("\nTesting single-token decode...")
next_id = mx.array([[6]], dtype=mx.int32)
pos = mx.array([[5]], dtype=mx.int32)
start = time.time()
logits2, caches2 = model(next_id, pos, caches)
mx.eval(logits2)
elapsed2 = time.time() - start
print(f"Decode: {elapsed2:.3f}s")
print(f"Logits shape: {logits2.shape}")

print("\nALL TESTS PASSED")
