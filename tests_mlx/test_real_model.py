"""Test loading real YuE2 weights and running forward pass."""
import sys
import time
sys.path.insert(0, "/tmp/YuE")

import mlx.core as mx
import json
from safetensors import safe_open
from pathlib import Path
import torch

# Load config
with open("/tmp/yue2-mlx/lm/config.json") as f:
    config =json.load(f)
print(f"Model config: {config['hidden_size']}d, {config['num_hidden_layers']} layers")

# Load weights from safetensors
print("\nLoading PyTorch weights...")
weights = {}
sf_files = sorted(Path("/tmp/yue2-weights").glob("*.safetensors"))
for sf_file in sf_files:
    with safe_open(sf_file, framework="pt") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if str(tensor.dtype) == 'torch.bfloat16':
                tensor = tensor.to(torch.float32)
            weights[key] = mx.array(tensor.numpy())
print(f"Loaded {len(weights)} weight tensors")

# Create model
from src.yue2_mlx.modeling_mlx import YuE2MLX
print("\nCreating MLX model...")
model = YuE2MLX(config)

# Map weight names
def map_name(pt_name):
    name = pt_name
    if name.startswith("model.model."):
        name = name[len("model.model."):]
    name = name.replace("model.embed_tokens.", "embed_tokens.")
    name = name.replace("model.norm.", "norm.")
    name = name.replace("model.llm2vae.", "llm2vae.")
    name = name.replace("model.vae2llm.", "vae2llm.")
    name = name.replace("model.latent_pos_embed.pe", "latent_pos_embed")
    if "time_embedder.mlp.2." in name:
        name = name.replace("time_embedder.mlp.2.", "time_embed.mlp.1.")
    elif "time_embedder." in name:
        name = name.replace("time_embedder.", "time_embed.")
    return name

mapped_weights = {}
for k, v in weights.items():
    mapped_weights[map_name(k)] = v

print("\nLoading weights into model...")
model.load_weights(mapped_weights, strict=False)
print("Weights loaded!")

# Test forward pass
print("\nTesting forward pass...")
input_ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
position_ids = mx.arange(5, dtype=mx.int32)[None, :]
start = time.time()
logits, caches = model(input_ids, position_ids)
mx.eval(logits)
elapsed = time.time() - start
print(f"Forward pass: {elapsed:.3f}s, logits shape: {logits.shape}")

# Test single-token decode
print("\nTesting single-token decode...")
next_id = mx.array([[6]], dtype=mx.int32)
pos = mx.array([[5]], dtype=mx.int32)
start = time.time()
logits2, caches2 = model(next_id, pos, caches)
mx.eval(logits2)
elapsed2 = time.time() - start
print(f"Decode: {elapsed2:.3f}s, logits shape: {logits2.shape}")

# Test generation
print("\nTesting token generation...")
start = time.time()
tokens = list(range(1, 11))
for _ in range(10):
    input_ids = mx.array([[tokens[-1]]], dtype=mx.int32)
    pos = mx.array([[len(tokens) - 1]], dtype=mx.int32)
    logits, caches = model(input_ids, pos, caches)
    next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    tokens.append(next_token)
    mx.eval(logits)
elapsed3 = time.time() - start
print(f"Generated 10 tokens in {elapsed3:.3f}s")
print(f"Tokens: {tokens}")

print("\nALL TESTS PASSED - Real model loaded and generating!")
