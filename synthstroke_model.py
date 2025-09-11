"""
SynthStroke model wrapper for Hugging Face Hub integration.
"""
import torch
from monai.networks.nets import UNet
from huggingface_hub import PyTorchModelHubMixin
import json
from typing import Dict, Any, Optional


class SynthStrokeModel(torch.nn.Module, PyTorchModelHubMixin):
    """
    SynthStroke model for stroke lesion segmentation.
    
    This model uses a UNet architecture and can be trained with synthetic data
    for robust stroke lesion segmentation across different imaging protocols.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()

        # Default configuration
        self.config = {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 2,  # Background + Lesion
            "channels": [32, 64, 128, 256, 320, 320],
            "strides": [2, 2, 2, 2, 2],
            "kernel_size": 3,
            "up_kernel_size": 3,
            "num_res_units": 1,
            "act": "PRELU",
            "norm": "INSTANCE",
            "dropout": 0.0,
            "bias": True,
            "adn_ordering": "NDA",
            "model_type": "baseline"  # "baseline", "synthstroke", or "multibrain"
        }

        if config is not None:
            self.config.update(config)

        # Validate configuration
        self._validate_config()

        # Create the UNet model
        self.unet = UNet(
            spatial_dims=self.config["spatial_dims"],
            in_channels=self.config["in_channels"],
            out_channels=self.config["out_channels"],
            channels=self.config["channels"],
            strides=self.config["strides"],
            kernel_size=self.config["kernel_size"],
            up_kernel_size=self.config["up_kernel_size"],
            num_res_units=self.config["num_res_units"],
            act=self.config["act"],
            norm=self.config["norm"],
            dropout=self.config["dropout"],
            bias=self.config["bias"],
            adn_ordering=self.config["adn_ordering"],
        )

    def _validate_config(self):
        """Validate model configuration parameters."""
        required_keys = ["spatial_dims", "in_channels", "out_channels", "channels"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")

        if self.config["spatial_dims"] != 3:
            raise ValueError("Only 3D models are supported")

        if self.config["in_channels"] not in [1, 4]:
            raise ValueError(f"Unsupported number of input channels: {self.config['in_channels']}. Must be 1 or 4.")

        if self.config["out_channels"] not in [2, 6]:
            raise ValueError(f"Unsupported number of output channels: {self.config['out_channels']}. Must be 2 or 6.")
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.unet(x)
    
    def predict_segmentation(self, x, apply_softmax: bool = True, use_tta: bool = False):
        """
        Predict segmentation masks from input images with optional Test-Time Augmentation.

        Args:
            x: Input tensor of shape (batch_size, 1, H, W, D)
            apply_softmax: Whether to apply softmax to get probabilities
            use_tta: Whether to use flip-based Test-Time Augmentation

        Returns:
            Segmentation probabilities or logits
        """
        # Input validation
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input tensor (batch, channels, H, W, D), got {x.dim()}D")

        if x.shape[1] != self.config["in_channels"]:
            raise ValueError(f"Expected {self.config['in_channels']} input channels, got {x.shape[1]}")

        # Ensure input is on the same device as the model
        if hasattr(self, 'device'):
            x = x.to(self.device)
        else:
            # Try to infer device from model parameters
            device = next(self.parameters()).device
            x = x.to(device)

        self.eval()
        with torch.no_grad():
            if use_tta:
                return self._predict_with_tta(x, apply_softmax)
            else:
                logits = self.forward(x)
                if apply_softmax:
                    return torch.softmax(logits, dim=1)
                return logits

    def _predict_with_tta(self, x, apply_softmax: bool = True):
        """
        Perform flip-based Test-Time Augmentation for improved predictions.

        Args:
            x: Input tensor of shape (batch_size, 1, H, W, D)
            apply_softmax: Whether to apply softmax to get probabilities

        Returns:
            Averaged segmentation predictions from all flip augmentations
        """
        batch_size, channels, H, W, D = x.shape
        device = x.device

        # Store predictions from all augmentations
        all_predictions = []

        # Original prediction
        logits_original = self.forward(x)
        all_predictions.append(logits_original)

        # Define flip transformations for 3D volumes
        # flips: (flip_H, flip_W, flip_D)
        flip_configs = [
            (True, False, False),   # Flip along H axis (axial)
            (False, True, False),   # Flip along W axis (coronal)
            (False, False, True),   # Flip along D axis (sagittal)
            (True, True, False),    # Flip H and W
            (True, False, True),    # Flip H and D
            (False, True, True),    # Flip W and D
            (True, True, True),     # Flip all axes
        ]

        for flip_H, flip_W, flip_D in flip_configs:
            # Apply flips
            x_flipped = x.clone()

            if flip_H:
                x_flipped = torch.flip(x_flipped, dims=[2])  # Flip H dimension
            if flip_W:
                x_flipped = torch.flip(x_flipped, dims=[3])  # Flip W dimension
            if flip_D:
                x_flipped = torch.flip(x_flipped, dims=[4])  # Flip D dimension

            # Forward pass
            logits_flipped = self.forward(x_flipped)

            # Undo the flips on the predictions to align with original orientation
            if flip_H:
                logits_flipped = torch.flip(logits_flipped, dims=[2])
            if flip_W:
                logits_flipped = torch.flip(logits_flipped, dims=[3])
            if flip_D:
                logits_flipped = torch.flip(logits_flipped, dims=[4])

            all_predictions.append(logits_flipped)

        # Average all predictions (original + 7 flipped versions = 8 total)
        averaged_logits = torch.mean(torch.stack(all_predictions), dim=0)

        if apply_softmax:
            return torch.softmax(averaged_logits, dim=1)
        return averaged_logits
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Load model from a .pt checkpoint file.
        
        Args:
            checkpoint_path: Path to the .pt checkpoint file
            config: Optional configuration dictionary
            
        Returns:
            SynthStrokeModel instance
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Extract model state dict
        if "net" in checkpoint:
            state_dict = checkpoint["net"]
        else:
            state_dict = checkpoint
        
        # Create model instance
        model = cls(config=config)
        model.load_state_dict(state_dict)
        
        return model
    
    def get_config(self):
        """Return model configuration."""
        return self.config.copy()

    def get_model_info(self):
        """Return detailed information about the model."""
        return {
            "model_type": self.config["model_type"],
            "input_channels": self.config["in_channels"],
            "output_channels": self.config["out_channels"],
            "spatial_dims": self.config["spatial_dims"],
            "architecture": "3D UNet",
            "framework": "MONAI",
            "tta_support": True,
            "parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

    def to(self, device):
        """Move model to specified device."""
        super().to(device)
        # Store device for later use
        self.device = device
        return self

    @property
    def num_classes(self):
        """Get number of output classes."""
        return self.config["out_channels"]

    @property
    def input_shape(self):
        """Get expected input shape (excluding batch dimension)."""
        return (self.config["in_channels"], -1, -1, -1)  # H, W, D can vary
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """Save model and configuration to directory."""
        # Import safetensors for safe model saving
        try:
            from safetensors.torch import save_file
            use_safetensors = True
        except ImportError:
            use_safetensors = False
            print("⚠️  safetensors not available, falling back to pickle. Install with: pip install safetensors")

        # Save model weights
        if use_safetensors:
            # Save in safetensors format (secure)
            state_dict = self.state_dict()
            safetensors_path = f"{save_directory}/model.safetensors"
            save_file(state_dict, safetensors_path)
            print(f"✅ Saved model weights in safetensors format: {safetensors_path}")
        else:
            # Fallback to PyTorchModelHubMixin default (includes pickle warning)
            super().save_pretrained(save_directory, **kwargs)

        # Save configuration
        config_path = f"{save_directory}/config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        """Load model from Hugging Face Hub."""
        from huggingface_hub import hf_hub_download
        import os

        # First download the config file
        try:
            config_path = hf_hub_download(model_id, "config.json")
            with open(config_path, "r") as f:
                config = json.load(f)
        except:
            config = cls().config  # Use default config if download fails

        # Create model instance with the loaded config
        model = cls(config=config)

        # Load the model weights using PyTorchModelHubMixin
        model = super().from_pretrained(model_id, **kwargs)

        return model


# Convenience functions for different model types
def create_baseline_model():
    """Create baseline model configuration (real ATLAS T1w data only)."""
    config = {
        "out_channels": 2,  # Background, Stroke
        "model_type": "baseline"
    }
    return SynthStrokeModel(config=config)


def create_synth_model():
    """Create Synth model configuration (synthetic data from OASIS3 + ATLAS)."""
    config = {
        "out_channels": 6,  # Background, GM, WM, GM/WM PV, CSF, Stroke
        "model_type": "synth"
    }
    return SynthStrokeModel(config=config)


def create_synth_pseudo_model():
    """Create SynthPseudo model configuration (synthetic + pseudo-labels)."""
    config = {
        "out_channels": 6,  # Background, GM, WM, GM/WM PV, CSF, Stroke
        "model_type": "synth_pseudo"
    }
    return SynthStrokeModel(config=config)


def create_synth_plus_model():
    """Create SynthPlus model configuration (synthetic + real multi-dataset)."""
    config = {
        "out_channels": 6,  # Background, GM, WM, GM/WM PV, CSF, Stroke
        "model_type": "synth_plus"
    }
    return SynthStrokeModel(config=config)


def create_qatlas_model():
    """Create qATLAS model configuration (synthetic qMRI from ATLAS)."""
    config = {
        "out_channels": 2,  # Background, Stroke
        "model_type": "qatlas",
        "in_channels": 1  # T1-weighted MRI (qMRI used for data generation)
    }
    return SynthStrokeModel(config=config)


def create_qsynth_model():
    """Create qSynth model configuration (qMRI-constrained synthetic data)."""
    config = {
        "out_channels": 6,  # Background, GM, WM, GM/WM PV, CSF, Stroke
        "model_type": "qsynth",
        "in_channels": 1  # T1-weighted MRI (qMRI used for data generation)
    }
    return SynthStrokeModel(config=config)


# Legacy functions for backward compatibility
def create_synthstroke_model():
    """Create SynthStroke model configuration (legacy function)."""
    return create_synth_model()


def create_multibrain_model():
    """Create MultiBrain model configuration (legacy function)."""
    return create_synth_model()