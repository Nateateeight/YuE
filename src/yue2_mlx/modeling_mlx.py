"""YuE2 AR–NAR Mixture-of-Transformers in MLX."""
from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
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
    # x: [B, T, H, D], cos/sin: [B, T, half] -> [B, T, 1, half] for broadcast over heads
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

        # Transpose to [B, H, T, D] for attention computation
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
        # out: [B, H, T_q, D] where T_q is query tokens
        # For decode with cache, we only want the output for the new tokens (last T positions)
        T_q = out.shape[2]
        out = out[:, :, -T:, :]  # Keep only the last T tokens (the new ones)
        out = out.transpose(0, 2, 1, 3)  # [B, T, H, D]
        H, D = out.shape[2], out.shape[3]
        out = out.reshape(B, T, -1)
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
        B, T, _ = x.shape
        q = attn.q_norm(attn.q_proj(x).reshape(B, T, attn.num_heads, attn.head_dim))
        k = attn.k_norm(attn.k_proj(x).reshape(B, T, attn.num_kv_heads, attn.head_dim))
        v = attn.v_proj(x).reshape(B, T, attn.num_kv_heads, attn.head_dim)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)
        return q, k, v

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
        max_frames = config.get("max_latent_frames", 24576)
        pe = mx.zeros((max_frames, config["hidden_size"]))
        pos = mx.arange(max_frames, dtype=mx.float32)[:, None]
        div = mx.exp(mx.arange(0, config["hidden_size"], 2, dtype=mx.float32) * (-math.log(10000.0) / config["hidden_size"]))
        pe[:, 0::2] = mx.sin(pos * div)
        pe[:, 1::2] = mx.cos(pos * div)
        self.latent_pos_embed = pe

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
        return self.norm(x), new_caches

    def __call__(self, *args, **kwargs):
        return self.forward_ar(*args, **kwargs)

    @staticmethod
    def from_pretrained(path: str) -> 'YuE2MLX':
        path = Path(path)
        with open(path / "config.json") as f:
            config = json.load(f)
        model = YuE2MLX(config)
        weights = mx.load(str(path / "weights.npz"))
        model.load_weights(weights, strict=False)
        return model
