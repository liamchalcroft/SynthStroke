"""
SynthStroke model wrapper for Hugging Face Hub integration.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal, Tuple, Union
from dataclasses import dataclass, field

import torch
import numpy as np
import monai as mn
from monai.networks.nets import UNet
from monai.data import MetaTensor
from safetensors.torch import save_file, load_file
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download


class ModelConstants:
    """Constants used throughout the model."""
    DEFAULT_PATCH_SIZE = 128
    SLIDING_WINDOW_OVERLAP = 0.5
    GAUSSIAN_SIGMA_SCALE = 0.125
    SW_BATCH_SIZE = 1
    NUMERICAL_EPSILON = 1e-8
    STROKE_CLASS_MULTICLASS = 5
    STROKE_CLASS_BINARY = 1
    CT_MIN_HU = 0
    CT_MAX_HU = 80
    DEFAULT_MC_SAMPLES = 10
    VOXEL_SPACING_MM = 1.0
    
    TTA_FLIP_CONFIGS = [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ]


@dataclass
class ModelConfig:
    """Type-safe configuration for SynthStroke models."""
    spatial_dims: Literal[3] = 3
    in_channels: Literal[1, 4] = 1
    out_channels: Literal[2, 6] = 2
    channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 320, 320])
    strides: List[int] = field(default_factory=lambda: [2, 2, 2, 2, 2])
    kernel_size: int = 3
    up_kernel_size: int = 3
    num_res_units: int = 1
    act: str = "PRELU"
    norm: str = "INSTANCE"
    dropout: float = 0.0
    bias: bool = True
    adn_ordering: str = "NDA"
    model_type: str = "baseline"
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        if self.spatial_dims != 3:
            raise ValueError("Only 3D models are supported")
        if self.in_channels not in [1, 4]:
            raise ValueError(f"Unsupported input channels: {self.in_channels}. Must be 1 or 4.")
        if self.out_channels not in [2, 6]:
            raise ValueError(f"Unsupported output channels: {self.out_channels}. Must be 2 or 6.")
        if len(self.channels) == 0:
            raise ValueError("channels list cannot be empty")
        if len(self.strides) != len(self.channels) - 1:
            raise ValueError(f"strides length ({len(self.strides)}) must be channels length - 1")
        if not (0 <= self.dropout < 1):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.kernel_size < 1 or self.up_kernel_size < 1:
            raise ValueError("kernel sizes must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "spatial_dims": self.spatial_dims,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "channels": self.channels,
            "strides": self.strides,
            "kernel_size": self.kernel_size,
            "up_kernel_size": self.up_kernel_size,
            "num_res_units": self.num_res_units,
            "act": self.act,
            "norm": self.norm,
            "dropout": self.dropout,
            "bias": self.bias,
            "adn_ordering": self.adn_ordering,
            "model_type": self.model_type
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        return cls(**config_dict)


@dataclass
class InferenceConfig:
    """Configuration for inference parameters."""
    apply_softmax: bool = True
    use_tta: bool = False
    use_sliding_window: bool = True
    patch_size: int = ModelConstants.DEFAULT_PATCH_SIZE
    use_mc_dropout: bool = False
    mc_samples: int = ModelConstants.DEFAULT_MC_SAMPLES
    
    def __post_init__(self):
        if self.patch_size < 1:
            raise ValueError("patch_size must be positive")
        if self.mc_samples < 1:
            raise ValueError("mc_samples must be positive")


@dataclass
class PredictionResult:
    """Standard prediction result container."""
    prediction: torch.Tensor
    probabilities: Optional[torch.Tensor] = None
    lesion_mask: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MCPredictionResult(PredictionResult):
    """Monte Carlo dropout prediction result with uncertainty estimates."""
    mean_prediction: torch.Tensor = None
    prediction_std: torch.Tensor = None
    epistemic_uncertainty: torch.Tensor = None
    predictive_entropy: Optional[torch.Tensor] = None
    all_samples: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        if self.mean_prediction is not None and self.prediction is None:
            self.prediction = self.mean_prediction


class SynthStrokeError(Exception):
    """Base exception class for SynthStroke model errors."""
    pass


class SynthStrokeModel(torch.nn.Module, PyTorchModelHubMixin):
    """
    SynthStroke model for stroke lesion segmentation.
    
    This model uses a UNet architecture and can be trained with synthetic data
    for robust stroke lesion segmentation across different imaging protocols.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self._device_cache: Optional[torch.device] = None
        self._sliding_window_cache: Dict[int, mn.inferers.SlidingWindowInferer] = {}
        
        self.unet = UNet(
            spatial_dims=self.config.spatial_dims,
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            channels=self.config.channels,
            strides=self.config.strides,
            kernel_size=self.config.kernel_size,
            up_kernel_size=self.config.up_kernel_size,
            num_res_units=self.config.num_res_units,
            act=self.config.act,
            norm=self.config.norm,
            dropout=self.config.dropout,
            bias=self.config.bias,
            adn_ordering=self.config.adn_ordering,
        )

    @property
    def device(self) -> torch.device:
        """Get the device this model is on."""
        if self._device_cache is None:
            self._device_cache = next(self.parameters()).device
        return self._device_cache
    
    def to(self, device: Union[torch.device, str]) -> 'SynthStrokeModel':
        """Move model to specified device."""
        result = super().to(device)
        result._device_cache = None
        result._sliding_window_cache.clear()
        return result
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.unet(x)
    
    def enable_mc_dropout(self) -> None:
        """Enable Monte Carlo dropout."""
        for module in self.modules():
            if isinstance(module, (torch.nn.Dropout3d, torch.nn.Dropout)):
                module.train()
    
    def disable_mc_dropout(self) -> None:
        """Disable Monte Carlo dropout."""
        self.eval()
    
    def _get_sliding_window_inferer(self, patch_size: int) -> mn.inferers.SlidingWindowInferer:
        """Get or create a cached sliding window inferer."""
        if patch_size not in self._sliding_window_cache:
            self._sliding_window_cache[patch_size] = mn.inferers.SlidingWindowInferer(
                roi_size=[patch_size, patch_size, patch_size],
                sw_batch_size=ModelConstants.SW_BATCH_SIZE,
                overlap=ModelConstants.SLIDING_WINDOW_OVERLAP,
                mode="gaussian",
                sigma_scale=ModelConstants.GAUSSIAN_SIGMA_SCALE,
                cval=0.0,
                sw_device=None,
                device=None,
                progress=False,
                cache_roi_weight_map=False,
            )
        return self._sliding_window_cache[patch_size]
    
    def _apply_flip_transforms(self, x: torch.Tensor, flip_dims: List[int]) -> torch.Tensor:
        """Apply flip transforms."""
        result = x
        for dim in flip_dims:
            result = torch.flip(result, dims=[dim])
        return result
    
    def _generate_tta_predictions(self, x: torch.Tensor, inference_fn: callable) -> torch.Tensor:
        """Generate TTA predictions with streaming aggregation."""
        running_sum = inference_fn(x)
        count = 1
        
        for flip_config in ModelConstants.TTA_FLIP_CONFIGS:
            flip_dims = []
            if flip_config[0]:
                flip_dims.append(2)
            if flip_config[1]:
                flip_dims.append(3)
            if flip_config[2]:
                flip_dims.append(4)
            
            if not flip_dims:
                continue
                
            x_flipped = self._apply_flip_transforms(x, flip_dims)
            pred_flipped = inference_fn(x_flipped)
            pred_unflipped = self._apply_flip_transforms(pred_flipped, flip_dims)
            running_sum = running_sum + pred_unflipped
            count += 1
        
        return running_sum / count
    
    def _predict_with_mc_dropout(self, x: torch.Tensor, inference_fn: callable, 
                                 num_samples: int) -> Dict[str, torch.Tensor]:
        """Perform Monte Carlo dropout inference using Welford's algorithm."""
        self.enable_mc_dropout()
        
        running_mean = None
        running_m2 = None
        
        for i in range(num_samples):
            with torch.inference_mode():
                sample_pred = inference_fn(x)
                
                if running_mean is None:
                    running_mean = sample_pred.clone()
                    running_m2 = torch.zeros_like(sample_pred)
                else:
                    delta = sample_pred - running_mean
                    running_mean += delta / (i + 1)
                    delta2 = sample_pred - running_mean
                    running_m2 += delta * delta2
        
        self.disable_mc_dropout()
        
        variance = running_m2 / num_samples
        std = torch.sqrt(variance + ModelConstants.NUMERICAL_EPSILON)
        
        predictive_entropy = None
        if torch.all((running_mean >= 0) & (running_mean <= 1)):
            entropy = -torch.sum(
                running_mean * torch.log(running_mean + ModelConstants.NUMERICAL_EPSILON), 
                dim=1, keepdim=True
            )
            predictive_entropy = entropy
        
        return {
            'mean': running_mean,
            'std': std,
            'epistemic_uncertainty': variance,
            'predictive_entropy': predictive_entropy
        }
    
    def _needs_sliding_window(self, x: torch.Tensor, patch_size: int) -> bool:
        """Check if sliding window inference is needed.

        Sliding window is needed if:
        1. Any dimension exceeds patch_size, OR
        2. Any dimension is not divisible by 32 (UNet downsampling factor)
        """
        _, _, H, W, D = x.shape
        divisor = 32  # 2^5 for 5 stride-2 downsampling layers
        size_exceeds = H > patch_size or W > patch_size or D > patch_size
        not_divisible = (H % divisor != 0) or (W % divisor != 0) or (D % divisor != 0)
        return size_exceeds or not_divisible
    
    def _predict_with_sliding_window(self, x: torch.Tensor, apply_softmax: bool,
                                    use_tta: bool, patch_size: int) -> torch.Tensor:
        """Perform sliding window inference."""
        window = self._get_sliding_window_inferer(patch_size)
        all_predictions = []

        for i in range(x.shape[0]):
            img = x[i:i+1]

            if use_tta:
                def inference_fn(input_tensor):
                    # Keep batch dimension for TTA flip operations
                    return window(input_tensor, self)
                pred = self._generate_tta_predictions(img, inference_fn)
            else:
                pred = window(img, self)

            all_predictions.append(pred)

        predictions = torch.cat(all_predictions, dim=0)
        return torch.softmax(predictions, dim=1) if apply_softmax else predictions
    
    def _predict_with_tta(self, x: torch.Tensor, apply_softmax: bool) -> torch.Tensor:
        """Perform Test-Time Augmentation."""
        def inference_fn(input_tensor):
            logits = self.forward(input_tensor)
            return torch.softmax(logits, dim=1) if apply_softmax else logits
        
        return self._generate_tta_predictions(x, inference_fn)
    
    def _run_inference(self, x: torch.Tensor, config: InferenceConfig) -> torch.Tensor:
        """Run inference based on configuration."""
        self.eval()
        # Use inference_mode for better CPU performance (faster than no_grad)
        with torch.inference_mode():
            if config.use_sliding_window and self._needs_sliding_window(x, config.patch_size):
                return self._predict_with_sliding_window(
                    x, config.apply_softmax, config.use_tta, config.patch_size
                )
            elif config.use_tta:
                return self._predict_with_tta(x, config.apply_softmax)
            else:
                logits = self.forward(x)
                return torch.softmax(logits, dim=1) if config.apply_softmax else logits
    
    def _create_lesion_mask(self, prediction: torch.Tensor) -> torch.Tensor:
        """Create lesion mask based on model configuration."""
        if self.config.out_channels == 6:
            return (prediction == ModelConstants.STROKE_CLASS_MULTICLASS).float()
        else:
            return (prediction == ModelConstants.STROKE_CLASS_BINARY).float()
    
    def _create_prediction_result(self, prediction: torch.Tensor, x: torch.Tensor, 
                                 config: InferenceConfig) -> PredictionResult:
        """Create a standard prediction result."""
        discrete_pred = torch.argmax(prediction, dim=1) if config.apply_softmax else prediction
        
        return PredictionResult(
            prediction=discrete_pred,
            probabilities=prediction if config.apply_softmax else None,
            lesion_mask=self._create_lesion_mask(discrete_pred),
            metadata={
                'input_shape': list(x.shape),
                'patch_size': config.patch_size,
                'used_tta': config.use_tta,
                'used_sliding_window': config.use_sliding_window,
                'model_type': self.config.model_type
            }
        )
    
    def _predict_comprehensive_mc(self, x: torch.Tensor, config: InferenceConfig) -> Dict[str, torch.Tensor]:
        """Comprehensive MC dropout inference."""
        def inference_strategy_fn(input_tensor):
            if config.use_sliding_window and self._needs_sliding_window(input_tensor, config.patch_size):
                return self._predict_with_sliding_window(
                    input_tensor, config.apply_softmax, config.use_tta, config.patch_size
                )
            elif config.use_tta:
                return self._predict_with_tta(input_tensor, config.apply_softmax)
            else:
                logits = self.forward(input_tensor)
                return torch.softmax(logits, dim=1) if config.apply_softmax else logits
        
        return self._predict_with_mc_dropout(x, inference_strategy_fn, config.mc_samples)
    
    def _create_mc_prediction_result(self, mc_results: Dict[str, torch.Tensor], 
                                     x: torch.Tensor, config: InferenceConfig) -> MCPredictionResult:
        """Create an MC prediction result."""
        mean_pred = mc_results['mean']
        discrete_pred = torch.argmax(mean_pred, dim=1) if config.apply_softmax else mean_pred
        
        return MCPredictionResult(
            prediction=discrete_pred,
            mean_prediction=mean_pred,
            probabilities=mean_pred if config.apply_softmax else None,
            lesion_mask=self._create_lesion_mask(discrete_pred),
            prediction_std=mc_results['std'],
            epistemic_uncertainty=mc_results['epistemic_uncertainty'],
            predictive_entropy=mc_results.get('predictive_entropy'),
            metadata={
                'input_shape': list(x.shape),
                'patch_size': config.patch_size,
                'used_tta': config.use_tta,
                'used_sliding_window': config.use_sliding_window,
                'mc_samples': config.mc_samples,
                'model_type': self.config.model_type
            }
        )
    
    def _validate_and_prepare_input(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Validate and prepare input tensor."""
        # Handle nibabel image objects
        if hasattr(x, 'get_fdata'):
            x = x.get_fdata()

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor, np.ndarray, or nibabel image, got {type(x).__name__}. "
                "If using nibabel, pass nib.load(path).get_fdata() or the nibabel object directly."
            )

        if x.dim() == 3:
            # Add channel and batch dimensions: (H,W,D) -> (1,1,H,W,D)
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 4:
            # Add batch dimension: (C,H,W,D) -> (1,C,H,W,D)
            x = x.unsqueeze(0)
        elif x.dim() != 5:
            raise ValueError(f"Expected 3D, 4D or 5D tensor, got {x.dim()}D")

        if x.shape[1] != self.config.in_channels:
            raise ValueError(f"Expected {self.config.in_channels} channels, got {x.shape[1]}")

        return x.to(self.device)
    
    def predict_comprehensive(self, x: Union[torch.Tensor, np.ndarray], 
                             config: Optional[InferenceConfig] = None) -> Union[PredictionResult, MCPredictionResult]:
        """
        Comprehensive prediction method combining all inference techniques.
        
        Args:
            x: Input tensor or numpy array
            config: Inference configuration (defaults to InferenceConfig())
            
        Returns:
            PredictionResult or MCPredictionResult
        """
        if config is None:
            config = InferenceConfig()
        
        x = self._validate_and_prepare_input(x)
        
        if config.use_mc_dropout:
            mc_results = self._predict_comprehensive_mc(x, config)
            return self._create_mc_prediction_result(mc_results, x, config)
        else:
            pred = self._run_inference(x, config)
            return self._create_prediction_result(pred, x, config)
    
    def predict_segmentation(self, x: Union[torch.Tensor, np.ndarray], 
                           apply_softmax: bool = True, use_tta: bool = False, 
                           patch_size: int = ModelConstants.DEFAULT_PATCH_SIZE, 
                           use_sliding_window: bool = True) -> torch.Tensor:
        """Legacy prediction method for backward compatibility."""
        config = InferenceConfig(
            apply_softmax=apply_softmax,
            use_tta=use_tta,
            patch_size=patch_size,
            use_sliding_window=use_sliding_window,
            use_mc_dropout=False
        )
        return self.predict_comprehensive(x, config).prediction
    
    @staticmethod
    def preprocess_image(image_data: Union[np.ndarray, torch.Tensor], 
                        affine: Optional[np.ndarray] = None, 
                        is_ct: bool = False, 
                        device: Optional[Union[torch.device, str]] = None) -> Tuple[torch.Tensor, mn.transforms.Compose]:
        """Apply preprocessing pipeline."""
        if isinstance(image_data, np.ndarray):
            image_tensor = torch.from_numpy(image_data).float()
        else:
            image_tensor = image_data.float()
        
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        if device is not None:
            image_tensor = image_tensor.to(device)
        
        if affine is not None:
            image_tensor = MetaTensor(image_tensor, affine=affine)
        
        batch = {"img": image_tensor}
        
        ct_clip = mn.transforms.LambdaD(
            keys=["img"], 
            func=lambda x: torch.clamp(x, ModelConstants.CT_MIN_HU, ModelConstants.CT_MAX_HU), 
            allow_missing_keys=True
        ) if is_ct else mn.transforms.Identity()
        
        preproc = mn.transforms.Compose([
            mn.transforms.EnsureChannelFirstD(keys=["img"], allow_missing_keys=True),
            mn.transforms.ToTensorD(keys=["img"], device=device, allow_missing_keys=True),
            ct_clip,
            mn.transforms.OrientationD(keys=["img"], axcodes="RAS", allow_missing_keys=True),
            mn.transforms.SpacingD(keys=["img"], pixdim=ModelConstants.VOXEL_SPACING_MM, allow_missing_keys=True),
            mn.transforms.HistogramNormalizeD(keys="img"),
            mn.transforms.NormalizeIntensityD(keys="img", nonzero=False, channel_wise=True, allow_missing_keys=True),
        ])
        
        batch = preproc(batch)
        return batch["img"], preproc
    
    def predict_comprehensive_with_preprocessing(self, image_data, affine=None, is_ct: bool = False, 
                                               apply_softmax: bool = True, use_tta: bool = False, 
                                               use_sliding_window: bool = True, patch_size: int = 128, 
                                               use_mc_dropout: bool = False, mc_samples: int = 10):
        """Complete inference pipeline with preprocessing."""
        device = self.device
        
        preprocessed_img, preproc_transforms = self.preprocess_image(
            image_data, affine, is_ct, device
        )
        
        if preprocessed_img.dim() == 4:
            preprocessed_img = preprocessed_img.unsqueeze(0)
        
        inference_config = InferenceConfig(
            apply_softmax=apply_softmax,
            use_tta=use_tta,
            use_sliding_window=use_sliding_window,
            patch_size=patch_size,
            use_mc_dropout=use_mc_dropout,
            mc_samples=mc_samples
        )
        
        results = self.predict_comprehensive(preprocessed_img, config=inference_config)
        
        if isinstance(results, MCPredictionResult):
            probabilities = results.mean_prediction
            prediction_std = results.prediction_std
            epistemic_uncertainty = results.epistemic_uncertainty
            predictive_entropy = results.predictive_entropy
        else:
            probabilities = results.probabilities if results.probabilities is not None else results.prediction
            prediction_std = None
            epistemic_uncertainty = None
            predictive_entropy = None
        
        prediction = torch.argmax(probabilities, dim=1)
        
        lesion_mask = (prediction == 5).float() if self.config.out_channels == 6 else (prediction == 1).float()
        
        outputs_to_transform = {
            'probabilities': probabilities,
            'prediction': prediction,
            'lesion_mask': lesion_mask
        }
        
        if prediction_std is not None:
            outputs_to_transform['prediction_std'] = prediction_std
        if epistemic_uncertainty is not None:
            outputs_to_transform['epistemic_uncertainty'] = epistemic_uncertainty
        if predictive_entropy is not None:
            outputs_to_transform['predictive_entropy'] = predictive_entropy
        
        if hasattr(preproc_transforms, 'inverse') and hasattr(preprocessed_img, 'applied_operations'):
            for key, tensor in outputs_to_transform.items():
                if tensor is not None:
                    tensor.applied_operations = preprocessed_img.applied_operations
            
            with mn.transforms.utils.allow_missing_keys_mode(preproc_transforms):
                for key in outputs_to_transform:
                    if outputs_to_transform[key] is not None:
                        inverse_result = preproc_transforms.inverse({"img": outputs_to_transform[key]})
                        outputs_to_transform[key] = inverse_result["img"]
        
        final_results = {}
        for key, tensor in outputs_to_transform.items():
            if tensor is not None:
                final_results[key] = tensor.cpu().numpy() if hasattr(tensor, 'cpu') else tensor
        
        if isinstance(results, MCPredictionResult) and results.all_samples is not None:
            final_results['mc_samples'] = results.all_samples.cpu().numpy() if hasattr(results.all_samples, 'cpu') else results.all_samples
        
        return final_results
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> 'SynthStrokeModel':
        """Load model from a local safetensors checkpoint file."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_dir = checkpoint_path.parent
        
        # Try to load config.json from same directory first
        config_path = checkpoint_dir / "config.json"
        
        # If not found, try subdirectory with same name as checkpoint stem
        if not config_path.exists():
            subdir_config = checkpoint_dir / checkpoint_path.stem / "config.json"
            if subdir_config.exists():
                config_path = subdir_config
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = ModelConfig.from_dict(config_dict)
        
        # Load model weights
        state_dict = load_file(str(checkpoint_path))
        
        # Adjust state dict keys if needed
        if not any(key.startswith("unet.") for key in state_dict.keys()):
            state_dict = {f"unet.{key}": value for key, value in state_dict.items()}
        
        model = cls(config=config)
        model.load_state_dict(state_dict)
        return model
    
    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> 'SynthStrokeModel':
        """Load model from Hugging Face Hub."""
        # Load config.json from Hub
        config_path = hf_hub_download(model_id, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = ModelConfig.from_dict(config_dict)
        
        # Download and load model weights
        safetensors_path = hf_hub_download(model_id, "model.safetensors")
        state_dict = load_file(safetensors_path)
        
        # Adjust state dict keys if needed
        if not any(key.startswith("unet.") for key in state_dict.keys()):
            state_dict = {f"unet.{key}": value for key, value in state_dict.items()}
        
        model = cls(config=config)
        model.load_state_dict(state_dict)
        return model
    
    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        """Save model and configuration to directory."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        state_dict = self.state_dict()
        safetensors_path = save_directory / "model.safetensors"
        save_file(state_dict, str(safetensors_path))
        
        # Save configuration
        config_path = save_directory / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    def get_config(self) -> ModelConfig:
        """Return model configuration."""
        return self.config
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return detailed information about the model."""
        return {
            "model_type": self.config.model_type,
            "input_channels": self.config.in_channels,
            "output_channels": self.config.out_channels,
            "spatial_dims": self.config.spatial_dims,
            "architecture": "3D UNet",
            "framework": "MONAI",
            "tta_support": True,
            "mc_dropout_support": True,
            "sliding_window_support": True,
            "parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    @property
    def num_classes(self) -> int:
        """Get number of output classes."""
        return self.config.out_channels
    
    @property
    def input_shape(self) -> Tuple[int, int, int, int]:
        """Get expected input shape (excluding batch dimension)."""
        return (self.config.in_channels, -1, -1, -1)


# Convenience functions for different model types
def create_baseline_model():
    """Create baseline model configuration."""
    config = ModelConfig(out_channels=2, model_type="baseline")
    return SynthStrokeModel(config=config)


def create_synth_model():
    """Create Synth model configuration."""
    config = ModelConfig(out_channels=6, model_type="synth")
    return SynthStrokeModel(config=config)


def create_synth_pseudo_model():
    """Create SynthPseudo model configuration."""
    config = ModelConfig(out_channels=6, model_type="synth_pseudo")
    return SynthStrokeModel(config=config)


def create_synth_plus_model():
    """Create SynthPlus model configuration."""
    config = ModelConfig(out_channels=6, model_type="synth_plus")
    return SynthStrokeModel(config=config)


def create_qatlas_model():
    """Create qATLAS model configuration."""
    config = ModelConfig(out_channels=2, model_type="qatlas")
    return SynthStrokeModel(config=config)


def create_qsynth_model():
    """Create qSynth model configuration."""
    config = ModelConfig(out_channels=6, model_type="qsynth")
    return SynthStrokeModel(config=config)
