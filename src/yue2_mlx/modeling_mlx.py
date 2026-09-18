"""YuE2 AR–NAR Mixture-of-Transformers in MLX."""
from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any
import json
import math

import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones(dim)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        var = mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True)
        return x * mx.rsqrt(var + self.eps) * self.weight.astype(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 1000000.0):
        super().__init__()
        self.head_dim = head_dim
        self.base = base

    def __call__(self, position_ids: mx.array) -> Tuple[mx.array, mx.array]:
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.base ** (mx.arange(0, self.head_dim, 2, dtype=mx.float32) / self.head_dim))
        angles = position_ids.astype(mx.float32)[:, :, None] * inv_freq[None, None, :]
        return mx.cos(angles), mx.sin(angles)


def _apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    cos = cos[:, :, None, :]
    sin = sin[:, :, None, :]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos.astype(x.dtype)
    sin = sin.astype(x.dtype)
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


class Attention(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]

        self.q_proj = nn.Linear(config["hidden_size"], self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config["hidden_size"], self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config["hidden_size"], self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config["hidden_size"], bias=False)
        self.q_norm = RMSNorm(self.head_dim, config["rms_norm_eps"])
        self.k_norm = RMSNorm(self.head_dim, config["rms_norm_eps"])

    def project_qkv(self, x: mx.array, cos: mx.array, sin: mx.array):
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim))
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)
        return q, k, v

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array,
                 cache: Optional[Tuple[mx.array, mx.array]] = None,
                 attention_mask: Optional[mx.array] = None,
                 is_causal: bool = False) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim))
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        groups = self.num_heads // self.num_kv_heads
        if groups > 1:
            k = mx.repeat(k, groups, axis=1)
            v = mx.repeat(v, groups, axis=1)

        if cache is not None:
            prev_k, prev_v = cache
            k = mx.concatenate([prev_k, k], axis=2)
            v = mx.concatenate([prev_v, v], axis=2)
        new_cache = (k, v)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = q @ k.transpose(0, 1, 3, 2) * scale
        if attention_mask is not None:
            attn = attn + attention_mask
        if is_causal:
            seq_len = attn.shape[-1]
            mask = mx.triu(mx.full((seq_len, seq_len), -1e9), k=1)
            attn = attn + mask
        attn = mx.softmax(attn, axis=-1)
        out = attn @ v
        T_q = out.shape[2]
        out = out[:, :, -T:, :].transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.o_proj(out), new_cache


class MLP(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.gate_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.input_layernorm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.self_attn = Attention(config)
        self.nar_input_layernorm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.nar_self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.mlp = MLP(config)
        self.nar_pre_mlp_layernorm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.nar_mlp = MLP(config)

    def _project_qkv(self, attn: Attention, x: mx.array, cos: mx.array, sin: mx.array):
        return attn.project_qkv(x, cos, sin)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array,
                 ar_mask: Optional[mx.array] = None,
                 cache: Optional[Tuple[mx.array, mx.array]] = None,
                 attention_mask: Optional[mx.array] = None,
                 is_causal: bool = False
                 ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        if ar_mask is not None:
            mask_3d = ar_mask[:, :, None]
            ln_ar = self.input_layernorm(x)
            ln_nar = self.nar_input_layernorm(x)
            q_ar, k_ar, v_ar = self._project_qkv(self.self_attn, ln_ar, cos, sin)
            q_nar, k_nar, v_nar = self._project_qkv(self.nar_self_attn, ln_nar, cos, sin)

            query = mx.where(mask_3d[:, :, None, :], q_ar, q_nar)
            key = mx.where(mask_3d[:, :, None, :], k_ar, k_nar)
            value = mx.where(mask_3d[:, :, None, :], v_ar, v_nar)

            query = query.transpose(0, 2, 1, 3)
            key = key.transpose(0, 2, 1, 3)
            value = value.transpose(0, 2, 1, 3)

            groups = self.self_attn.num_heads // self.self_attn.num_kv_heads
            if groups > 1:
                key = mx.repeat(key, groups, axis=1)
                value = mx.repeat(value, groups, axis=1)

            if cache is not None:
                prev_k, prev_v = cache
                key = mx.concatenate([prev_k, key], axis=2)
                value = mx.concatenate([prev_v, value], axis=2)
            new_cache = (key, value)

            scale = 1.0 / math.sqrt(self.self_attn.head_dim)
            attn = query @ key.transpose(0, 1, 3, 2) * scale
            if attention_mask is not None:
                attn = attn + attention_mask
            attn = mx.softmax(attn, axis=-1)
            out = (attn @ value).transpose(0, 2, 1, 3)
            B, S = x.shape[:2]
            out = out.reshape(B, S, -1)

            o_ar = self.self_attn.o_proj(out)
            o_nar = self.nar_self_attn.o_proj(out)
            h = mx.where(mask_3d, o_ar, o_nar)
            x = x + h

            ar_out = self.mlp(self.post_attention_layernorm(x))
            nar_out = self.nar_mlp(self.nar_pre_mlp_layernorm(x))
            x = x + mx.where(mask_3d, ar_out, nar_out)
            return x, new_cache
        else:
            h, new_cache = self.self_attn(self.input_layernorm(x), cos, sin,
                                          cache, attention_mask, is_causal)
            x = x + h
            x = x + self.mlp(self.post_attention_layernorm(x))
            return x, new_cache


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        layers = [nn.Linear(frequency_embedding_size, hidden_size),
                  nn.Linear(hidden_size, hidden_size)]
        self.mlp = layers

    def __call__(self, t: mx.array) -> mx.array:
        half = self.frequency_embedding_size // 2
        freqs = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)
        if t.ndim == 0:
            t = t.reshape(1)
        args = t[:, None] * freqs[None, :]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        x = self.mlp[0](emb)
        x = nn.silu(x)
        return self.mlp[1](x)


class YuE2MLX(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = [DecoderLayer(config) for _ in range(config["num_hidden_layers"])]
        self.norm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.rotary = RotaryEmbedding(config["head_dim"], config["rope_theta"])
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        self.llm2vae = nn.Linear(config["hidden_size"], config["latent_dim"])
        self.vae2llm = nn.Linear(config["latent_dim"], config["hidden_size"])
        self.time_embed = TimestepEmbedder(config["hidden_size"])

    def forward_ar(self, input_ids: mx.array, position_ids: mx.array,
                   caches: Optional[List] = None, attention_mask: Optional[mx.array] = None,
                   ar_mask: Optional[mx.array] = None) -> Tuple[mx.array, List]:
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary(position_ids)
        if caches is None:
            caches = [None] * len(self.layers)
        new_caches = []
        for i, layer in enumerate(self.layers):
            is_causal = (input_ids.shape[1] == 1) and i == 0
            x, new_cache = layer(x, cos, sin, ar_mask, caches[i], attention_mask, is_causal)
            new_caches.append(new_cache)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_caches

    def forward_nar(self, input_ids: mx.array, position_ids: mx.array,
                    x_t: mx.array, t_value: float,
                    ar_mask: Optional[mx.array] = None) -> mx.array:
        """NAR velocity prediction for flow matching."""
        B, S = input_ids.shape
        
        # AR token embeddings
        token_emb = self.embed_tokens(input_ids)
        
        # Build latent hidden for NAR positions
        x_nar = mx.concatenate([mx.zeros((1, 64)), x_t, mx.zeros((1, 64))], axis=0)
        t_shifted = self._shift_t_value(t_value)
        
        nar_hidden = self.vae2llm(x_nar[None])
        t_embed = self.time_embed(t_shifted)
        nar_hidden = nar_hidden + t_embed[None, :, :]
        
        nar_length = x_nar.shape[0]
        pe = self._compute_pe(nar_length)
        nar_hidden = nar_hidden + pe[None]
        
        nar_pos_ids = mx.arange(S, S + nar_length, dtype=mx.int32)[None]
        
        full_emb = mx.concatenate([token_emb, nar_hidden], axis=1)
        
        full_ar_mask = mx.concatenate([
            mx.ones((B, S), dtype=mx.bool_),
            mx.zeros((B, nar_length), dtype=mx.bool_)
        ], axis=1)
        
        full_pos_ids = mx.concatenate([position_ids, nar_pos_ids], axis=1)
        cos_full, sin_full = self.rotary(full_pos_ids)
        
        attn_mask = self._hybrid_attention_mask(S, nar_length)
        
        x = full_emb
        for i, layer in enumerate(self.layers):
            x, _ = layer(x, cos_full, sin_full, full_ar_mask, None, attn_mask, False)
        
        x = self.norm(x)
        
        nar_content_start = S + 1
        nar_content_end = S + nar_length - 1
        nar_out = x[:, nar_content_start:nar_content_end, :]
        
        v_pred = self.llm2vae(nar_out)
        return v_pred[0]

    def _compute_pe(self, length: int) -> mx.array:
        """Compute sinusoidal position embeddings on the fly."""
        pe = mx.zeros((length, self.config["hidden_size"]))
        pos = mx.arange(length, dtype=mx.float32)[:, None]
        div = mx.exp(mx.arange(0, self.config["hidden_size"], 2, dtype=mx.float32) * 
                     (-math.log(10000.0) / self.config["hidden_size"]))
        pe[:, 0::2] = mx.sin(pos * div)
        pe[:, 1::2] = mx.cos(pos * div)
        return pe

    def _shift_t_value(self, t_value: float) -> mx.array:
        t_sig = mx.sigmoid(mx.array(t_value, dtype=mx.float32))
        shift = self.config.get("timestep_shift", 1.0)
        result = shift * t_sig / (1 + (shift - 1) * t_sig)
        return result.reshape(1)

    def _hybrid_attention_mask(self, ar_len: int, nar_len: int) -> mx.array:
        S = ar_len + nar_len
        
        ar_q = mx.zeros((1, S), dtype=mx.float32)
        ar_q[:, :ar_len] = 1.0
        ar_k = mx.zeros((1, S), dtype=mx.float32)
        ar_k[:, :ar_len] = 1.0
        nar_q = mx.zeros((1, S), dtype=mx.float32)
        nar_q[:, ar_len:] = 1.0
        nar_k = mx.zeros((1, S), dtype=mx.float32)
        nar_k[:, ar_len:] = 1.0
        causal = mx.tril(mx.ones((S, S)))
        
        mask = (ar_q.T @ ar_k * causal) + (nar_q.T @ ar_k) + (nar_q.T @ nar_k)
        attn_mask = mx.where(mask > 0, mx.array(0.0), mx.array(-1e9))
        return attn_mask[None, :, :]

    def __call__(self, *args, **kwargs):
        return self.forward_ar(*args, **kwargs)
