"""MLX port of YuE2 Oobleck VAE: encoder + decoder with SnakeBeta and weight_norm.

The Oobleck encoder uses weight_norm (which we pre-absorb into weights at load).
SnakeBeta activation: x + (1/beta) * sin(alpha*x)^2
ResidualUnit → EncoderBlock/DecoderBlock → full encoder/decoder.

MLX provides mx.conv1d and mx.conv_transpose1d; weight_norm is done offline.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import math

import mlx.core as mx
import mlx.nn as nn


class SnakeBeta(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # Vanilla: alpha=beta=1.0, trainable
        self.alpha = mx.zeros(channels)
        self.beta = mx.zeros(channels)
        self.eps = 1e-9

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, C, T], alpha/beta: [C] → [1, C, 1]
        alpha = mx.exp(self.alpha)[None, :, None]
        beta = mx.exp(self.beta)[None, :, None]
        return x + (1.0 / (beta + self.eps)) * mx.sin(alpha * x) ** 2


class WeightNormConv1d(nn.Module):
    """Conv1d layer. Accepts PyTorch-style input (B, C, L) internally transposing for MLX.

    MLX expects (B, L, C_in) and weight (C_out, K, C_in).
    We accept (B, C_in, L) and produce (B, C_out, L).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        scale = math.sqrt(2.0 / (in_channels * kernel_size))
        weight = mx.random.normal((out_channels, kernel_size, in_channels)) * scale
        self.weight = weight
        if bias:
            self.bias = mx.zeros(out_channels)
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, L) → (B, L, C)
        x = x.transpose(0, 2, 1)
        out = mx.conv1d(x, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation)
        # out: (B, L, C_out) → (B, C_out, L)
        out = out.transpose(0, 2, 1)
        if self.bias is not None:
            out = out + self.bias[:, None]
        return out


class WeightNormConvTranspose1d(nn.Module):
    """ConvTranspose1d layer. Accepts (B, C, L) → (B, C_out, L)."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.stride = stride
        self.padding = padding
        scale = math.sqrt(2.0 / (out_channels * kernel_size))
        weight = mx.random.normal((out_channels, kernel_size, in_channels)) * scale
        self.weight = weight
        if bias:
            self.bias = mx.zeros(out_channels)
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        x = x.transpose(0, 2, 1)  # (B, C, L) → (B, L, C)
        out = mx.conv_transpose1d(x, self.weight, stride=self.stride, padding=self.padding)
        out = out.transpose(0, 2, 1)  # (B, L, C_out) → (B, C_out, L)
        if self.bias is not None:
            out = out + self.bias[:, None]
        return out


def activation(act_type: str, channels: int = None):
    if act_type == "snake":
        return SnakeBeta(channels)
    elif act_type == "elu":
        return nn.ELU()
    elif act_type == "none":
        return nn.Identity()
    raise ValueError(f"Unknown activation: {act_type}")


class ResidualUnit(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, act_type):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        layers = [
            activation(act_type, out_ch),
            WeightNormConv1d(in_ch, out_ch, kernel_size=7, dilation=dilation, padding=padding),
            activation(act_type, out_ch),
            WeightNormConv1d(out_ch, out_ch, kernel_size=1),
        ]
        self.layers = layers

    def __call__(self, x):
        residual = x
        for layer in self.layers:
            x = layer(x)
        return x + residual


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act_type):
        super().__init__()
        layers = [
            ResidualUnit(in_ch, in_ch, 1, act_type),
            ResidualUnit(in_ch, in_ch, 3, act_type),
            ResidualUnit(in_ch, in_ch, 9, act_type),
            activation(act_type, in_ch),
            WeightNormConv1d(in_ch, out_ch, kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2)),
        ]
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act_type):
        super().__init__()
        layers = [
            activation(act_type, in_ch),
            WeightNormConvTranspose1d(in_ch, out_ch, kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2)),
            ResidualUnit(out_ch, out_ch, 1, act_type),
            ResidualUnit(out_ch, out_ch, 3, act_type),
            ResidualUnit(out_ch, out_ch, 9, act_type),
        ]
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class OobleckEncoder(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        in_ch = config.get("in_channels", 2)
        channels = config.get("channels", 64)
        c_mults = [1] + list(config.get("c_mults", [1, 2, 4, 8]))
        strides = config.get("strides", [2, 4, 8, 8])
        use_snake = config.get("use_snake", False)
        latent_dim = config.get("latent_dim", 32)
        act_type = "snake" if use_snake else "elu"

        layers = [WeightNormConv1d(in_ch, c_mults[0]*channels, kernel_size=7, padding=3)]
        for i in range(len(c_mults) - 1):
            layers.append(EncoderBlock(c_mults[i]*channels, c_mults[i+1]*channels, strides[i], act_type))
        layers.extend([
            activation(act_type, c_mults[-1]*channels),
            WeightNormConv1d(c_mults[-1]*channels, latent_dim, kernel_size=3, padding=1),
        ])
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class OobleckDecoder(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        out_ch = config.get("out_channels", 2)
        channels = config.get("channels", 64)
        c_mults = [1] + list(config.get("c_mults", [1, 2, 4, 8]))
        strides = config.get("strides", [2, 4, 8, 8])
        use_snake = config.get("use_snake", False)
        latent_dim = config.get("latent_dim", 32)
        final_tanh = config.get("final_tanh", True)
        act_type = "snake" if use_snake else "elu"

        layers = [WeightNormConv1d(latent_dim, c_mults[-1]*channels, kernel_size=7, padding=3)]
        for i in range(len(c_mults) - 1, 0, -1):
            layers.append(DecoderBlock(c_mults[i]*channels, c_mults[i-1]*channels, strides[i-1], act_type))
        layers.extend([
            activation(act_type, c_mults[0]*channels),
            WeightNormConv1d(c_mults[0]*channels, out_ch, kernel_size=7, padding=3, bias=False),
        ])
        if final_tanh:
            layers.append(nn.Tanh())
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class YuE2VAEMLX(nn.Module):
    def __init__(self, encoder_config: Dict[str, Any], decoder_config: Dict[str, Any],
                 decoder_only: bool = False):
        super().__init__()
        self.decoder_only = decoder_only
        if not decoder_only:
            self.encoder = OobleckEncoder(encoder_config)
        self.decoder = OobleckDecoder(decoder_config)

    def encode(self, audio: mx.array) -> mx.array:
        pre = self.encoder(audio)
        mean = pre[:, :pre.shape[1]//2, :]
        scale = pre[:, pre.shape[1]//2:, :]
        stdev = nn.softplus(scale) + 1e-4
        return mean

    def decode(self, latent: mx.array) -> mx.array:
        return self.decoder(latent)
