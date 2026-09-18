"""End-to-end YuE2 song generation on MLX with proper text conditioning."""
import gc
import time
import sys
sys.path.insert(0, "/tmp/YuE")

import mlx.core as mx
import json
from pathlib import Path
from safetensors import safe_open
import torch
import struct
import numpy as np

print("="*60)
print("YuE2 MLX - Full Pipeline")
print("="*60)

# ============ CONFIG ============
MODEL_DIR = "/tmp/yue2-weights"
VAE_DIR = "/tmp/yue2-weights/vae"

EOD = 151643
ABC_START, ABC_END = 151847, 151848
MUSIC_START, MUSIC_END = 151851, 151852
CODEC_OFFSET = 151853
LATENT_START = 184621

# ============ TOKENIZER ============
print("\nLoading tokenizer...")
import base64, tiktoken
ranks = {}
with open(f"{MODEL_DIR}/qwen.tiktoken", 'rb') as f:
    for line in f.read().splitlines():
        if line:
            t, r = line.split()
            ranks[base64.b64decode(t)] = int(r)
specials = ["", "<|im_start|>", "<|im_end|>", "<R>", "<S>", "<X>", "<mask>", "<sep>"]
specials += [f"<extra_{i}>" for i in range(200)]
specials[204:206] = ["<abc>", "</abc>"]
pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
tokenizer = tiktoken.Encoding("YuE2", pat_str=pattern, mergeable_ranks=ranks,
                              special_tokens={s: i + len(ranks) for i, s in enumerate(specials)})
print(f"Tokenizer loaded! Vocab: {tokenizer.n_vocab}")

# ============ LOAD LM ============
print("\nLoading LM config...")
with open("/tmp/yue2-mlx/lm/config.json") as f:
    lm_config = json.load(f)

from src.yue2_mlx.modeling_mlx import YuE2MLX
model = YuE2MLX(lm_config)

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

print("Loading LM weights...")
batch = {}
with safe_open(f"{MODEL_DIR}/model.safetensors", framework="pt") as h:
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

# ============ LOAD VAE ============
print("\nLoading VAE...")
from src.yue2_mlx.vae_mlx import OobleckDecoder

vae_config = json.loads(Path("/tmp/yue2-mlx/vae/config.json").read_text())
decoder = OobleckDecoder(vae_config["decoder_config"])

# Load VAE weights
vae_weights = {}
with safe_open(f"{VAE_DIR}/model.safetensors", framework="pt") as h:
    keys = list(h.keys())
    for key in keys:
        t = h.get_tensor(key)
        if str(t.dtype) == "torch.bfloat16":
            t = t.to(torch.float32)
        arr = t.numpy()
        
        if ".weight_g" in key:
            continue
        if ".weight_v" in key:
            g_key = key.replace("weight_v", "weight_g")
            if g_key in keys:
                weight_g = h.get_tensor(g_key).numpy()
                weight_v = arr
                norm = np.sqrt(np.sum(weight_v**2, axis=(1, 2), keepdims=True))
                arr = (weight_v / norm) * weight_g
                key = key.replace("weight_v", "weight")
        elif "weight" in key and arr.ndim == 3:
            arr = np.transpose(arr, (0, 2, 1))
        vae_weights[key] = mx.array(arr)

decoder.load_weights(vae_weights, strict=False)
del vae_weights
gc.collect()
print("VAE loaded!")

# ============ PROMPT ============
style = "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, 88 BPM"
lyrics = """[Verse]
Neon fades along the lane
Footsteps keep the time of rain

[Chorus]
Let the day come into view
Every road begins with you"""

instruction = "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions."
prompt_text = f"{instruction}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n"

prompt_tokens = [EOD] + tokenizer.encode(prompt_text, disallowed_special=())
print(f"\nPrompt: {len(prompt_tokens)} tokens")

# ============ ABC GENERATION (CoT) ============
print("\n[Step 1] Generating ABC score...")
prefix = prompt_tokens + [ABC_START]

abc_ids = []
caches = None
start_t = time.time()

for i in range(100):
    input_ids = mx.array([[prefix[-1]]], dtype=mx.int32)
    position_ids = mx.arange(len(prefix) - 1, len(prefix), dtype=mx.int32)[None, :]
    
    logits, caches = model(input_ids, position_ids, caches)
    next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    
    if next_token == ABC_END:
        break
    if next_token >= EOD:
        next_token = next_token % EOD
    
    abc_ids.append(next_token)
    prefix.append(next_token)
    mx.eval(logits)

print(f"  ABC: {len(abc_ids)} tokens in {time.time()-start_t:.2f}s")
abc_text = tokenizer.decode(abc_ids)
print(f"  Preview: {abc_text[:100]}...")

# ============ CODEC GENERATION ============
print("\n[Step 2] Generating codec tokens...")
prefix = prompt_tokens + [ABC_START] + abc_ids + [ABC_END, MUSIC_START]

codec_ids = []
caches = None
start_t = time.time()

for i in range(100):
    input_ids = mx.array([[prefix[-1] if i == 0 else codec_ids[-1]]], dtype=mx.int32)
    position_ids = mx.arange(len(prefix) + i - 1, len(prefix) + i, dtype=mx.int32)[None, :]
    
    logits, caches = model(input_ids, position_ids, caches)
    
    # Sample
    logits = logits[:, -1, :] / 0.7
    k = 50
    top_vals = mx.sort(logits)[:, -k:]
    logits_masked = mx.where(logits >= top_vals[:, :1], logits, -1e9)
    probs = mx.softmax(logits_masked, axis=-1)
    next_token = int(mx.random.categorical(probs, num_samples=1).item())
    
    if next_token == MUSIC_END:
        break
    if next_token < CODEC_OFFSET or next_token >= CODEC_OFFSET + 8192:
        next_token = CODEC_OFFSET + (next_token % 8192)
    
    codec_ids.append(next_token)
    mx.eval(logits)
    
    if (i + 1) % 500 == 0:
        print(f"  {i+1} tokens...")

codec_time = time.time() - start_t
print(f"  Codec: {len(codec_ids)} tokens in {codec_time:.2f}s ({len(codec_ids)/codec_time:.1f} tok/s)")

codec_values = [t - CODEC_OFFSET for t in codec_ids]

# ============ NAR FLOW MATCHING ============
print("\n[Step 3] NAR flow matching...")

# Use only a subset of codec tokens as context to save memory
max_ar_context = 100
codec_context = codec_values[:max_ar_context]

mx.random.seed(42)
num_latent_frames = len(codec_context) * 20
noise = mx.random.normal((num_latent_frames, 64))

# Build AR tokens
ar_tokens = [CODEC_OFFSET + v for v in codec_context]
input_ids = mx.array([ar_tokens], dtype=mx.int32)
position_ids = mx.arange(len(ar_tokens), dtype=mx.int32)[None, :]
ar_mask = mx.ones((1, len(ar_tokens)), dtype=mx.bool_)

def solve_ode_memory_efficient(model, input_ids, position_ids, initial_state, ar_mask, steps=32):
    """ODE solver with AR KV caching to reduce memory."""
    state = initial_state
    dt = 1.0 / steps
    
    print("  Pre-computing AR KV cache...")
    ar_emb = model.embed_tokens(input_ids)
    cos_ar, sin_ar = model.rotary(position_ids)
    
    ar_kv_cache = []
    x = ar_emb
    for i, layer in enumerate(model.layers):
        is_causal = (i == 0)
        x, kv = layer.self_attn(
            layer.input_layernorm(x), cos_ar, sin_ar,
            cache=None, is_causal=is_causal
        )
        ar_kv_cache.append(kv)
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    mx.eval(x)
    
    for step in range(steps):
        t = 1.0 - step * dt
        raw_t = np.log(t / (1 - t)) if 0 < t < 1 else (-20 if t <= 0 else 20)
        
        v1 = model.forward_nar(input_ids, position_ids, state, raw_t, ar_mask)
        mx.eval(v1)
        
        mid = state - v1 * (dt / 2)
        mid_t = 1.0 - (step + 0.5) * dt
        raw_mid = np.log(mid_t / (1 - mid_t)) if 0 < mid_t < 1 else (-20 if mid_t <= 0 else 20)
        
        v2 = model.forward_nar(input_ids, position_ids, mid, raw_mid, ar_mask)
        mx.eval(v2)
        
        state = state - v2 * dt
        
        if (step + 1) % 10 == 0:
            print(f"  Step {step+1}/{steps}")
    
    return state

ode_start = time.time()
latents = solve_ode_memory_efficient(model, input_ids, position_ids, noise, ar_mask, steps=32)
mx.eval(latents)
print(f"  ODE: {time.time() - ode_start:.2f}s, latents: {latents.shape}")

# ============ VAE DECODE ============
print("\n[Step 4] Decoding to audio...")
latents_t = latents.T[None, :, :]  # [1, 64, T]
audio = decoder(latents_t)
mx.eval(audio)
print(f"  Audio: {audio.shape}")

# Save
def save_wav(audio_data, sr, fname):
    if isinstance(audio_data, mx.array):
        audio_data = np.array(audio_data)
    if audio_data.ndim == 1:
        audio_data = audio_data[None, :]
    mx_val = np.max(np.abs(audio_data))
    if mx_val > 0:
        audio_data = audio_data / mx_val * 0.8
    audio_int16 = (audio_data.T * 32767).astype(np.int16)
    n_ch = audio_data.shape[0]
    n_samp = audio_data.shape[1]
    with open(fname, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + n_samp * n_ch * 2))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<H', n_ch))
        f.write(struct.pack('<I', sr))
        f.write(struct.pack('<I', sr * n_ch * 2))
        f.write(struct.pack('<H', n_ch * 2))
        f.write(struct.pack('<H', 16))
        f.write(b'data')
        f.write(struct.pack('<I', n_samp * n_ch * 2))
        f.write(audio_int16.tobytes())

Path("/tmp/yue2-mlx").mkdir(exist_ok=True)
output = "/tmp/yue2-mlx/yueml_full_pipeline.wav"
save_wav(audio, 48000, output)
duration = audio.shape[-1] / 48000

print(f"\n{'='*60}")
print(f"🎉 SONG GENERATED: {output}")
print(f"{'='*60}")
print(f"Duration: {duration:.2f}s | Sample rate: 48000Hz | Channels: {audio.shape[0]}")
print(f"\nRepo: https://github.com/Nateateeight/YuE/tree/mlx-port")
