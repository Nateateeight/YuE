"""Convert YuE2 PyTorch weights to MLX format."""
import argparse
import json
from pathlib import Path
import torch
import mlx.core as mx
from safetensors import safe_open


def load_safetensors(directory):
    directory = Path(directory)
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors in {directory}")
    weights = {}
    for f in files:
        print(f"Loading {f.name}...")
        with safe_open(f, framework="pt") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                # Handle BFloat16 by casting to float32
                if str(tensor.dtype) == 'torch.bfloat16':
                    tensor = tensor.to(torch.float32)
                weights[key] = mx.array(tensor.numpy())
    return weights


def convert_weights(model_dir, vae_dir, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    lm_dir = output_path / "lm"
    vae_dir_out = output_path / "vae"
    lm_dir.mkdir(exist_ok=True)
    vae_dir_out.mkdir(exist_ok=True)

    # Copy configs
    model_config = json.loads(Path(model_dir, "config.json").read_text())
    vae_config = json.loads(Path(vae_dir, "config.json").read_text())
    with open(lm_dir / "config.json", "w") as f:
        json.dump(model_config, f, indent=2)
    with open(vae_dir_out / "config.json", "w") as f:
        json.dump(vae_config, f, indent=2)

    # Load LM weights
    print("Loading LM weights...")
    lm_weights = load_safetensors(model_dir)
    print(f"Saving {len(lm_weights)} LM weights...")
    mx.savez(str(lm_dir / "weights.npz"), **lm_weights)

    # Load VAE weights
    print("Loading VAE weights...")
    vae_weights = load_safetensors(vae_dir)
    print(f"Saving {len(vae_weights)} VAE weights...")
    mx.savez(str(vae_dir_out / "weights.npz"), **vae_weights)

    lm_size = (lm_dir / "weights.npz").stat().st_size / 1024 / 1024
    vae_size = (vae_dir_out / "weights.npz").stat().st_size / 1024 / 1024
    print(f"\nDone! LM: {lm_size:.1f} MB, VAE: {vae_size:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/tmp/yue2-weights")
    parser.add_argument("--vae-dir", default="/tmp/yue2-weights/vae")
    parser.add_argument("--output", default="/tmp/yue2-mlx")
    args = parser.parse_args()
    convert_weights(args.model_dir, args.vae_dir, args.output)
