"""Full YuE2 MLX song generation with proper conditioning."""
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
import base64
import tiktoken

print("="*60)
print("YuE2 MLX - Full Pipeline v2")
print("="*60)

# ============ CONFIG ============
MODEL_DIR = "/tmp/yue2-weights"
VAE_DIR = "/tmp/yue2-weights/vae"

EOD = 151643
ABC_START, ABC_END = 151847, 151848
MUSIC_START, MUSIC_END = 151851, 151852
CODEC_OFFSET = 151853
CODEC_SIZE = 8192

# ============ TOKENIZER ============
print("\nLoading tokenizer...")
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
print(f"Tokenizer: {tokenizer.n_vocab} vocab")

# ============ LOAD LM ============
print("\nLoading LM...")
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
style = "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, lyrical memorable melody, unhurried phrasing, 88 BPM"
lyrics = """[Verse]
Neon fades along the lane
Footsteps keep the time of rain
Fold the night and leave it here
Morning has a sky to clear

[Chorus]
Let the day come into view
Every road begins with you
Hold a little room for light
We will sing beyond the night"""

# Build prompt exactly as PyTorch version does
instruction = "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions."
prompt_text = f"{instruction}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n"
prompt_tokens = [EOD] + tokenizer.encode(prompt_text, disallowed_special=())
print(f"\nPrompt: {len(prompt_tokens)} tokens")

# ============ SAMPLING ============
def sample_logits(logits, temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=1.2, history=None):
    """Sample from logits with temperature, top-k, top-p, and repetition penalty."""
    if history and repetition_penalty != 1.0:
        recent = history[-50:] if len(history) > 50 else history
        for token_id in set(recent):
            if logits[0, token_id] < 0:
                logits[0, token_id] *= repetition_penalty
            else:
                logits[0, token_id] /= repetition_penalty
    
    if temperature == 0:
        return int(mx.argmax(logits, axis=-1).item())
    
    logits = logits / temperature
    
    # Top-k
    if top_k > 0:
        top_vals = mx.sort(logits)[:, -top_k:]
        logits = mx.where(logits >= top_vals[:, :1], logits, -1e9)
    
    # Top-p (nucleus sampling)
    if top_p < 1.0:
        probs = mx.softmax(logits, axis=-1)
        # Sort descending
        sorted_indices = mx.argsort(probs, axis=-1)[:, ::-1]
        sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
        cumsum = mx.cumsum(sorted_probs, axis=-1)
        # Create mask: remove tokens where cumsum - prob > top_p
        mask = cumsum - sorted_probs > top_p
        # Always keep top token
        mask = mx.concatenate([mx.zeros_like(mask[:, :1]), mask[:, 1:]], axis=1)
        # Apply mask
        sorted_probs = mx.where(mask, mx.array(0.0), sorted_probs)
        # Scatter back to original positions
        result = mx.zeros_like(probs)
        result = mx.put_along_axis(result, sorted_indices, sorted_probs, axis=-1)
        # Convert to log
        logits = mx.log(mx.maximum(result, 1e-10))
    
    probs = mx.softmax(logits, axis=-1)
    return int(mx.random.categorical(probs, num_samples=1).item())

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
    
    # Sample ABC token
    next_token = sample_logits(logits[:, -1, :], temperature=0.7, top_k=30, top_p=0.9, 
                              repetition_penalty=1.005, history=abc_ids)
    
    if next_token == ABC_END:
        break
    if next_token >= EOD:
        next_token = next_token % EOD
    
    abc_ids.append(next_token)
    prefix.append(next_token)
    mx.eval(logits)

print(f"  ABC: {len(abc_ids)} tokens in {time.time()-start_t:.2f}s")
abc_text = tokenizer.decode(abc_ids)
print(f"  ABC:\n{abc_text}")

# Save ABC for debugging
with open("/tmp/yue2-mlx/abc_score.txt", "w") as f:
    f.write(abc_text)
print(f"  Saved: /tmp/yue2-mlx/abc_score.txt")

# ============ SEMANTIC CODEC GENERATION ============
print("\n[Step 2] Generating semantic codec tokens...")
prefix = prompt_tokens + [ABC_START] + abc_ids + [ABC_END, MUSIC_START]

codec_ids = []
caches = None
start_t = time.time()

for i in range(200):
    input_ids = mx.array([[prefix[-1] if i == 0 else codec_ids[-1]]], dtype=mx.int32)
    position_ids = mx.arange(len(prefix) + i - 1, len(prefix) + i, dtype=mx.int32)[None, :]
    
    logits, caches = model(input_ids, position_ids, caches)
    
    # Sample codec token
    next_token = sample_logits(logits[:, -1, :], temperature=1.0, top_k=100, top_p=0.95,
                              repetition_penalty=1.2, history=codec_ids)
    
    if next_token == MUSIC_END:
        break
    
    # Keep in codec range
    if next_token < CODEC_OFFSET or next_token >= CODEC_OFFSET + CODEC_SIZE:
        next_token = CODEC_OFFSET + (next_token % CODEC_SIZE)
    
    codec_ids.append(next_token)
    mx.eval(logits)
    
    if (i + 1) % 50 == 0:
        print(f"  {i+1} tokens...")

codec_time = time.time() - start_t
print(f"  Codec: {len(codec_ids)} tokens in {codec_time:.2f}s ({len(codec_ids)/codec_time:.1f} tok/s)")

codec_values = [t - CODEC_OFFSET for t in codec_ids]

# ============ NAR FLOW MATCHING ============
print("\n[Step 3] NAR flow matching...")

# Use subset of codec tokens as context
max_ar_context = min(100, len(codec_context))
codec_context = codec_values[:max_ar_context]

mx.random.seed(42)
max_ar_context = min(100, len(codec_values))
codec_context = codec_values[:max_ar_context]
num_latent_frames = len(codec_context) * 20
noise = mx.random.normal((num_latent_frames, 64))

ar_tokens = [CODEC_OFFSET + v for v in codec_context]
input_ids = mx.array([ar_tokens], dtype=mx.int32)
position_ids = mx.arange(len(ar_tokens), dtype=mx.int32)[None, :]
ar_mask = mx.ones((1, len(ar_tokens)), dtype=mx.bool_)

def solve_ode(model, input_ids, position_ids, initial_state, ar_mask, steps=32):
    state = initial_state
    dt = 1.0 / steps
    
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
latents = solve_ode(model, input_ids, position_ids, noise, ar_mask, steps=32)
mx.eval(latents)
print(f"  ODE: {time.time() - ode_start:.2f}s, latents: {latents.shape}")

# ============ VAE DECODE ============
print("\n[Step 4] Decoding to audio...")
latents_t = latents.T[None, :, :]
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
output = "/tmp/yue2-mlx/yueml_full_pipeline_v2.wav"
save_wav(audio, 48000, output)
duration = audio.shape[-1] / 48000

print(f"\n{'='*60}")
print(f"🎉 SONG GENERATED: {output}")
print(f"{'='*60}")
print(f"Duration: {duration:.2f}s | Sample rate: 48000Hz | Channels: {audio.shape[0]}")
print(f"\nTimings:")
print(f"  ABC: {time.time()-start_t:.2f}s")
print(f"  Codec: {codec_time:.2f}s")
print(f"  ODE: {time.time() - ode_start:.2f}s")
print(f"\nRepo: https://github.com/Nateateeight/YuE/tree/mlx-port")
