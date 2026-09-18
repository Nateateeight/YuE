"""NAR flow matching ODE solver for YuE2 MLX.

Ported from PyTorch. Implements:
- song_chunks: splits codec tokens into overlapping context chunks
- CachedNAR: AR prefill once → cached KV → NAR velocity prediction
- 32-step midpoint ODE solver for acoustic latent generation
"""
from __future__ import annotations
from typing import Callable, Optional, List, Tuple
import math

import mlx.core as mx

# Token constants (from protocol.py)
CODEC_OFFSET = 1024
CODEC_SIZE = 8192
CONTEXT = 24576
LATENT_START = 184699
LATENT_END = 184700
MUSIC_END = 184702
ABC_END = 184703
EOD = 184701


def chunk_ranges(total_len: int, prefix_len: int, context: int) -> List[Tuple[int, int]]:
    """Compute (start, end) ranges for each chunk."""
    ranges = []
    start = 0
    while start < total_len:
        end = min(start + context - prefix_len, total_len)
        ranges.append((start, end))
        start = end
    return ranges


def song_chunks(prefix: List[int], codec: List[int], seed: int, context: int = CONTEXT):
    """Generate noise chunks for each codec chunk."""
    ranges = chunk_ranges(len(codec), len(prefix), context)
    # Use MLX random with seed
    mx.random.seed(seed)
    noise = mx.random.normal((len(codec), 64))
    chunks = []
    for a, b in ranges:
        chunk_prefix = prefix + [v + CODEC_OFFSET for v in codec[a:b]] + [MUSIC_END]
        chunks.append((chunk_prefix, noise[a:b]))
    return chunks


def nar_attention(q: mx.array, k: mx.array, v: mx.array, causal: bool = False) -> mx.array:
    """Grouped-query attention with optional causal mask.
    
    q, k, v: [T_q, H, D], [T_k, H, D], [T_k, H, D] or [T_k, H_kv, D] for GQA.
    """
    # Handle GQA: repeat k, v to match q heads
    if q.shape[1] != k.shape[1]:
        groups = q.shape[1] // k.shape[1]
        k = mx.repeat(k, groups, axis=1)
        v = mx.repeat(v, groups, axis=1)
    
    scale = 1.0 / math.sqrt(q.shape[-1])
    attn = q @ k.T * scale
    if causal:
        T = attn.shape[0]
        mask = mx.triu(mx.full((T, T), -1e9), k=1)
        attn = attn + mask
    attn = mx.softmax(attn, axis=-1)
    return attn @ v


class CachedNAR:
    """Cached NAR (Non-Autoregressive) flow matching predictor.
    
    Runs AR prefill once, caches KV states, then predicts velocity
    at each ODE step without re-running the AR tokens.
    """
    
    def __init__(self, model, chunk_codec: List[int], prefix: List[int], noise: mx.array):
        self.model = model
        self.noise = noise  # [T, 64]
        
        ar_length = len(chunk_codec)
        self.ar_length = ar_length
        
        # AR prefill: run the model on codec tokens, cache KV
        self.cache = []
        input_ids = mx.array([chunk_codec], dtype=mx.int32)
        position_ids = mx.arange(ar_length, dtype=mx.int32)[None, :]
        
        x = model.embed_tokens(input_ids)
        cos, sin = model.rotary(position_ids)
        
        for layer in model.layers:
            # Only run AR path for prefill
            x_in = layer.input_layernorm(x)
            q, k, v = layer.self_attn.project_qkv(x_in, cos, sin)
            h, new_cache = layer.self_attn(
                x_in, cos, sin,
                cache=None,  # No cache for prefill
                is_causal=True
            )
            x = x + h
            x = x + layer.mlp(layer.post_attention_layernorm(x))
            # Cache KV for NAR steps (store as [T, H, D])
            self.cache.append((new_cache[0][0], new_cache[1][0]))
    
    def velocity(self, state: mx.array, t: float) -> mx.array:
        """Predict velocity field v(x_t, t) for flow matching.
        
        state: [T_lat, 64] current ODE state
        t: timestep (raw, will be sigmoid-shifted)
        Returns: [T_lat, 64] predicted velocity
        """
        model = self.model
        
        # 1. Prepare NAR input: pad state with START/END tokens
        # x_nar: [T_lat + 2, 64] with zeros at START and END positions
        x_nar = mx.concatenate([
            mx.zeros((1, 64)),
            state,
            mx.zeros((1, 64))
        ], axis=0)
        
        # 2. Project to hidden: vae2llm(x_nar) + time_emb + pos_emb
        t_shifted = model._shift_t_value(t)
        x = model.vae2llm(x_nar[None])  # [1, T_lat+2, H]
        
        # Time embedding
        time_emb = model.time_embed(t_shifted)
        x = x + time_emb[None, :, :]  # broadcast over sequence
        
        # Position embedding
        pos_ids = mx.arange(x_nar.shape[0]).clamp(max=model.config["max_latent_frames"] - 1)
        pos_emb = model.latent_pos_embed[pos_ids]
        x = x + pos_emb[None, :, :]
        
        # 3. Forward through decoder layers (NAR path only)
        nar_length = x_nar.shape[0]
        cos, sin = model.rotary(mx.arange(nar_length, dtype=mx.int32)[None, :])
        
        for i, layer in enumerate(model.layers):
            ar_k, ar_v = self.cache[i]
            x_in = layer.nar_input_layernorm(x)
            q, k, v = layer.nar_self_attn.project_qkv(x_in, cos, sin)
            
            # Concatenate cached AR KV with current NAR KV
            k = mx.concatenate([ar_k[None, :, :], k[0]], axis=0)
            v = mx.concatenate([ar_v[None, :, :], v[0]], axis=0)
            
            h = nar_attention(q[0], k, v, causal=False)
            h = h.flatten(1)[None, :, :]  # [1, T, H*D]
            x = x + layer.nar_self_attn.o_proj(h)
            x = x + layer.nar_mlp(layer.nar_pre_mlp_layernorm(x))
        
        x = model.norm(x)
        # NAR head: predict velocity at content positions (skip START/END)
        v_pred = model.llm2vae(x)[0, 1:-1, :]  # [T_lat, 64]
        return v_pred
    
    def solve(self, steps: int = 32,
              cancelled: Optional[Callable[[], bool]] = None,
              on_progress: Optional[Callable[[int, int], None]] = None) -> mx.array:
        """32-step midpoint ODE solver.
        
        dx/dt = v(x_t, t)
        Uses midpoint method: x_{t+dt} = x_t - v(mid) * dt
        """
        state = self.noise
        dt = 1.0 / steps
        
        for step in range(steps):
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during flow matching")
            
            t = 1.0 - step * dt
            # Convert to raw logit space (sigmoid inverse)
            raw_t = math.log(t / (1 - t)) if 0 < t < 1 else (-20 if t <= 0 else 20)
            
            # First velocity evaluation
            first = self.velocity(state, raw_t)
            
            # Midpoint prediction
            mid = state - first * (dt / 2)
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during flow matching")
            
            # Midpoint velocity
            mid_t = 1.0 - (step + 0.5) * dt
            raw_mid = math.log(mid_t / (1 - mid_t)) if 0 < mid_t < 1 else (-20 if mid_t <= 0 else 20)
            
            # Full step
            state = state - self.velocity(mid, raw_mid) * dt
            
            if on_progress is not None:
                on_progress(step + 1, steps)
        
        return state


def generate_nar_latents(model, codec_tokens: List[int], seed: int = 42, context: int = CONTEXT) -> mx.array:
    """Generate acoustic latents for a full song using NAR flow matching.
    
    Args:
        model: YuE2MLX model
        codec_tokens: AR-generated semantic tokens
        random_seed: for noise generation
        context: chunk context size
    
    Returns:
        latents: [T, 64] acoustic latents
    """
    prefix = []  # Will be populated from AR generation
    
    chunks = song_chunks(prefix, codec_tokens, seed, context)
    all_latents = []
    
    for chunk_codec, noise in chunks:
        # For full CoT mode, prefix contains the score tokens
        # For now, assume codec_tokens is the full sequence and we process in chunks
        cached = CachedNAR(model, chunk_codec=[], prefix=[], noise=noise)
        latents = cached.solve(steps=32)
        all_latents.append(latents)
    
    return mx.concatenate(all_latents, axis=0)
