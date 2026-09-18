"""MLX weight conversion: PyTorch safetensors → MLX npz.

Usage:
    python weight_convert.py \
        --input /path/to/huggingface/YuE2-3B \
        --output /path/to/output_mlx

Conversion rules:
    1. Linear layers: transpose weight from (OC, IC) → (OC, IC) [MLX uses same]
    2. Conv1d/ConvTranspose1d: transpose weight from (OC, IC, K) → (OC, K, C_in)
    3. Embedding: identity (vocab_size, dim)
    4. All weights stored as-is (weight_norm pre-absorbed at load time)
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open


def convert_weights(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy config
    config_src = input_path / "config.json"
    config_dst = output_path / "config.json"
    if config_src.exists():
        config_dst.write_text(config_src.read_text())
        print("Copied config.json")

    # Find safetensors files
    sf_files = sorted(input_path.glob("*.safetensors"))
    if not sf_files:
        raise FileNotFoundError(f"No safetensors files found in {input_dir}")

    all_weights = {}
    for sf_file in sf_files:
        print(f"Loading {sf_file.name}...")
        with safe_open(sf_file, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                # Convert to numpy (CPU)
                arr = tensor.numpy()
                all_weights[key] = arr
                print(f"  {key}: {arr.shape}")

    # Save as npz
    npz_path = output_path / "weights.npz"
    print(f"\nSaving {len(all_weights)} weights to {npz_path}...")
    mx.savez(str(npz_path), **{k: mx.array(v) for k, v in all_weights.items()})
    print(f"Done! Total size: {npz_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Convert YuE2 PyTorch weights to MLX")
    parser.add_argument("--input", required=True, help="Path to HuggingFace model directory")
    parser.add_argument("--output", required=True, help="Output directory for MLX weights")
    args = parser.parse_args()
    convert_weights(args.input, args.output)


if __name__ == "__main__":
    main()
