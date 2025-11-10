#!/usr/bin/env python3
"""
Convert PyTorch .pt checkpoint files to secure safetensors format.

Note: The upload_to_hf.py script now automatically converts .pt files to safetensors
during the upload process, so manual conversion is typically not needed.

Usage:
    python convert_to_safetensors.py [--input_dir DIR] [--output_dir DIR]

Example:
    python convert_to_safetensors.py --input_dir checkpoints_for_hf --output_dir checkpoints_for_hf
"""

import argparse
import os
import torch
from pathlib import Path
from safetensors.torch import save_file


def convert_pt_to_safetensors(pt_path: str, safetensors_path: str):
    """Convert a single .pt file to .safetensors format."""
    print(f"Converting {pt_path} -> {safetensors_path}")

    # Load the checkpoint
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)

    # Extract state dict
    if "net" in checkpoint:
        state_dict = checkpoint["net"]
    else:
        state_dict = checkpoint

    # Import safetensors and save    
    save_file(state_dict, safetensors_path)
    print(f"Successfully converted to {safetensors_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch .pt files to safetensors format")
    parser.add_argument("--input_dir", type=str, default="checkpoints_for_hf",
                       help="Directory containing .pt files (default: checkpoints_for_hf)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save .safetensors files (default: same as input_dir)")
    parser.add_argument("--force", action="store_true",
                       help="Overwrite existing .safetensors files (default: False)")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .pt files
    pt_files = list(input_dir.glob("*.pt"))

    if not pt_files:
        print(f"Error: No .pt files found in {input_dir}")
        return

    print(f"Found {len(pt_files)} .pt files to convert:")
    for pt_file in pt_files:
        print(f"  - {pt_file.name}")

    # Convert each file
    converted_count = 0
    for pt_file in pt_files:
        safetensors_file = output_dir / f"{pt_file.stem}.safetensors"

        # Skip if output file exists and --force not used
        if safetensors_file.exists() and not args.force:
            print(f"Warning: Skipping {pt_file.name} (output file exists, use --force to overwrite)")
            continue

        try:
            convert_pt_to_safetensors(str(pt_file), str(safetensors_file))
            converted_count += 1
        except Exception as e:
            print(f"Error: Failed to convert {pt_file.name}: {e}")

    print(f"\nConversion complete! Converted {converted_count} files.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
