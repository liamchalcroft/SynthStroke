"""
Script to upload SynthStroke models to Hugging Face Hub.

This script uploads models to Hugging Face Hub using the existing README.md files
as model cards from the checkpoints_for_hf/ subdirectories.

Features:
- Automatically converts .pt files to secure safetensors format
- Uses existing README.md files as model cards
- Supports both single model and batch uploads
- Eliminates pickle security warnings on Hugging Face Hub

Single model upload:
    python upload_to_hf.py --checkpoint path/to/checkpoint.pt --model_name your-username/synthstroke-baseline --model_type baseline

Batch upload all models:
    python upload_to_hf.py --batch --username your-username

Supported model types:
    - baseline: 2-class segmentation (Background, Stroke)
    - synth: 6-class segmentation (Background, GM, WM, GM/WM PV, CSF, Stroke)
    - synth_pseudo: 6-class with pseudo-label augmentation
    - synth_plus: 6-class with multi-dataset augmentation
    - qatlas: 2-class qMRI-based segmentation
    - qsynth: 6-class qMRI-constrained segmentation

The script will automatically use the README.md file from the corresponding model
directory (e.g., checkpoints_for_hf/baseline/README.md) as the model card.
"""

import argparse
import os
import tempfile
from pathlib import Path
from synthstroke_model import SynthStrokeModel
from huggingface_hub import HfApi
from safetensors.torch import save_file


def convert_pt_to_safetensors(pt_path: str, safetensors_path: str):
    """
    Convert a PyTorch .pt checkpoint file to safetensors format.

    Args:
        pt_path: Path to the .pt checkpoint file
        safetensors_path: Path where to save the .safetensors file
    """
    import torch

    print(f"Converting {pt_path} to safetensors format...")

    # Load the checkpoint
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)

    # Extract state dict
    if "net" in checkpoint:
        state_dict = checkpoint["net"]
    else:
        state_dict = checkpoint

    # Save in safetensors format
    save_file(state_dict, safetensors_path)
    print(f"Converted to {safetensors_path}")


def copy_model_readme(checkpoints_dir: str, model_type: str, temp_dir: str):
    """Copy the README.md from the model directory to the temporary upload directory.

    Args:
        checkpoints_dir: Directory containing the checkpoints and model subdirectories
        model_type: Type of model (baseline, synth, etc.)
        temp_dir: Temporary directory where model files are saved for upload
    """
    # Map model types to their directory names
    model_dir_map = {
        "baseline": "baseline",
        "synth": "synth",
        "synth_pseudo": "synth_pseudo",
        "synth_plus": "synth_plus",
        "qatlas": "qatlas",
        "qsynth": "qsynth"
    }

    model_dir_name = model_dir_map.get(model_type, model_type)
    readme_source = os.path.join(checkpoints_dir, model_dir_name, "README.md")
    readme_dest = os.path.join(temp_dir, "README.md")

    if os.path.exists(readme_source):
        import shutil
        shutil.copy2(readme_source, readme_dest)
        print(f"Copied model card from {readme_source}")
    else:
        error_msg = f"README.md not found at {readme_source}! Cannot proceed without model card."
        print(f"Error: {error_msg}")
        raise FileNotFoundError(error_msg)    


def upload_all_models(checkpoints_dir: str = "checkpoints_for_hf", username: str = None, token: str = None):
    """
    Upload all models from the checkpoints directory to Hugging Face Hub.

    Args:
        checkpoints_dir: Directory containing checkpoints and model info
        username: Your Hugging Face username
        token: Hugging Face token
    """
    checkpoints_path = Path(checkpoints_dir)
    if not checkpoints_path.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    # Read model data
    model_data_file = checkpoints_path / "model_data.md"
    if not model_data_file.exists():
        raise FileNotFoundError(f"Model data file not found: {model_data_file}")

    # Parse model data to get model information
    models_info = {}
    current_model = None

    with open(model_data_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("## "):
                model_name = line[3:]
                # Convert model name to the format used in scripts
                if model_name == "Baseline":
                    current_model = "baseline"
                elif model_name == "Synth":
                    current_model = "synth"
                elif model_name == "SynthPseudo":
                    current_model = "synth_pseudo"
                elif model_name == "SynthPlus":
                    current_model = "synth_plus"
                elif model_name == "qATLAS":
                    current_model = "qatlas"
                elif model_name == "qSynth":
                    current_model = "qsynth"
                else:
                    current_model = model_name.lower()
                models_info[current_model] = {}
            elif current_model and line.startswith("file: "):
                models_info[current_model]["file"] = line[6:]
            elif current_model and line.startswith("name: "):
                models_info[current_model]["name"] = line[6:]

    print(f"Found {len(models_info)} models to upload:")
    for model_key, info in models_info.items():
        print(f"  - {model_key}: {info.get('name', 'Unknown')} ({info.get('file', 'Unknown')})")

    # Upload each model
    for model_key, info in models_info.items():
        checkpoint_file = info.get("file")
        if not checkpoint_file:
            print(f"Warning: Skipping {model_key}: No checkpoint file specified")
            continue

        checkpoint_path = checkpoints_path / checkpoint_file
        if not checkpoint_path.exists():
            print(f"Warning: Skipping {model_key}: Checkpoint file not found: {checkpoint_path}")
            continue

        if not username:
            print(f"Warning: Skipping {model_key}: No username provided")
            continue

        model_name = f"{username}/synthstroke-{model_key}"
        model_type = model_key

        print(f"\nUploading {model_key}...")
        upload_model(
            checkpoint_path=str(checkpoint_path),
            model_name=model_name,
            model_type=model_type,
            checkpoints_dir=str(checkpoints_path),
            token=token
        )
        print(f"Successfully uploaded {model_key}")


def upload_model(checkpoint_path: str, model_name: str, model_type: str, checkpoints_dir: str = "checkpoints_for_hf", token: str = None):
    """
    Upload a model to Hugging Face Hub.

    Args:
        checkpoint_path: Path to the checkpoint file (.pt or .safetensors). .pt files will be automatically converted to safetensors.
        model_name: Name for the model on HF Hub (e.g., "username/model-name")
        model_type: Type of model ("baseline", "synth", etc.)
        checkpoints_dir: Directory containing the model subdirectories with README.md files
        token: Hugging Face token (optional, can be set via huggingface-cli login)
    """

    checkpoint_path_obj = Path(checkpoint_path)
    
    # Handle .pt to .safetensors conversion if needed
    if checkpoint_path.endswith('.pt'):
        print(f"Detected .pt file: {checkpoint_path}")
        print("Converting to safetensors format for secure upload...")

        # Create temporary directory for safetensors + config.json
        conversion_temp_dir = tempfile.mkdtemp()
        temp_safetensors_path = os.path.join(conversion_temp_dir, f"{checkpoint_path_obj.stem}.safetensors")
        
        try:
            # Convert the .pt file to safetensors
            convert_pt_to_safetensors(checkpoint_path, temp_safetensors_path)
            
            # Copy config.json from model subdirectory to temp directory
            model_dir = Path(checkpoints_dir) / model_type
            config_source = model_dir / "config.json"
            if config_source.exists():
                import shutil
                shutil.copy2(config_source, os.path.join(conversion_temp_dir, "config.json"))
            
            checkpoint_to_use = temp_safetensors_path
            cleanup_conversion_dir = conversion_temp_dir
        except Exception as e:
            import shutil
            if os.path.exists(conversion_temp_dir):
                shutil.rmtree(conversion_temp_dir)
            raise RuntimeError(f"Failed to convert {checkpoint_path} to safetensors: {e}")
    else:
        # Already a safetensors file, use as-is
        checkpoint_to_use = checkpoint_path
        cleanup_conversion_dir = None

    print(f"Loading checkpoint from {checkpoint_to_use}...")

    # Load model from checkpoint (config.json will be loaded automatically)
    model = SynthStrokeModel.from_checkpoint(checkpoint_to_use)
    
    print(f"Uploading model to {model_name}...")
    
    # Create temporary directory for saving
    temp_dir = f"temp_{model_type}_model"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Save model with HF format
        model.save_pretrained(temp_dir)

        # Copy the model card (README.md) from the checkpoints directory
        copy_model_readme(checkpoints_dir, model_type, temp_dir)

        # List files that will be uploaded for verification
        print(f"Files to be uploaded from {temp_dir}:")
        import glob
        for file_path in glob.glob(f"{temp_dir}/*"):
            import os
            file_size = os.path.getsize(file_path)
            print(f"  - {os.path.basename(file_path)} ({file_size} bytes)")

        # Upload to Hub
        model.push_to_hub(model_name, token=token, use_temp_dir=False, local_files_only=False)

        print(f"Successfully uploaded model to https://huggingface.co/{model_name}")

    except Exception as e:
        print(f"Error: Failed to upload {model_type}: {str(e)}")
        raise e

    finally:
        # Clean up temporary directory
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        # Clean up temporary directory if it was created for conversion
        if cleanup_conversion_dir and os.path.exists(cleanup_conversion_dir):
            import shutil
            shutil.rmtree(cleanup_conversion_dir)
            print(f"Cleaned up temporary conversion directory: {cleanup_conversion_dir}")


def main():
    parser = argparse.ArgumentParser(description="Upload SynthStroke models to Hugging Face Hub")
    parser.add_argument("--batch", action="store_true",
                       help="Upload all models from checkpoints_for_hf directory")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints_for_hf",
                       help="Directory containing checkpoints (default: checkpoints_for_hf)")
    parser.add_argument("--username", type=str, default=None,
                       help="Your Hugging Face username (required for batch upload)")
    parser.add_argument("--checkpoint", type=str,
                       help="Path to the checkpoint file (.pt or .safetensors) (for single model upload)")
    parser.add_argument("--model_name", type=str,
                       help="Name for the model on HF Hub (e.g., 'username/model-name')")
    parser.add_argument("--model_type", type=str,
                       choices=["baseline", "synth", "synth_pseudo", "synth_plus", "qatlas", "qsynth"],
                       help="Type of model (for single model upload)")
    parser.add_argument("--token", type=str, default=None,
                       help="Hugging Face token (optional if already logged in)")

    args = parser.parse_args()

    if args.batch:
        # Batch upload mode
        if not args.username:
            parser.error("--username is required for batch upload")
        upload_all_models(
            checkpoints_dir=args.checkpoints_dir,
            username=args.username,
            token=args.token
        )
    else:
        # Single model upload mode
        if not args.checkpoint or not args.model_name or not args.model_type:
            parser.error("--checkpoint, --model_name, and --model_type are required for single model upload")

        # Verify checkpoint file exists
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

        # Upload the model
        upload_model(
            checkpoint_path=args.checkpoint,
            model_name=args.model_name,
            model_type=args.model_type,
            checkpoints_dir=args.checkpoints_dir,
            token=args.token
        )


if __name__ == "__main__":
    main()