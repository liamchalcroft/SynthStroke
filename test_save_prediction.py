"""
Test script to verify save_prediction function and dimensionality preservation
with sliding window and TTA.
"""
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from synthstroke_model import SynthStrokeModel, InferenceConfig, ModelConfig
import monai as mn


def load_test_image(image_path: str):
    """Load test image and return nibabel image object."""
    img = nib.load(image_path)
    return img


def verify_dimensionality(input_shape, output_shape, config_name: str):
    """Verify that output spatial dimensions match input."""
    # Input shape: (B, C, H, W, D) or (C, H, W, D)
    # Output shape: (B, C, H, W, D) or (C, H, W, D)
    
    # Extract spatial dimensions (last 3)
    input_spatial = input_shape[-3:]
    output_spatial = output_shape[-3:]
    
    if input_spatial == output_spatial:
        print(f"✓ {config_name}: Dimensionality preserved")
        print(f"  Input spatial: {input_spatial}, Output spatial: {output_spatial}")
        return True
    else:
        print(f"✗ {config_name}: Dimensionality mismatch!")
        print(f"  Input spatial: {input_spatial}, Output spatial: {output_spatial}")
        return False


def test_inference_configurations(model, image_data, affine, output_dir: Path, original_shape):
    """Test different inference configurations and verify dimensionality."""
    results = {}
    all_passed = True
    
    # Extract numpy array if nibabel object
    image_array = image_data.get_fdata() if hasattr(image_data, 'get_fdata') else image_data
    
    # Configuration 1: Basic inference with sliding window (no TTA)
    # Note: Large images typically need sliding window due to UNet downsampling requirements
    print("\n" + "="*60)
    print("Test 1: Basic inference with sliding window (no TTA)")
    print("="*60)
    config1 = InferenceConfig(
        apply_softmax=True,
        use_tta=False,
        use_sliding_window=True,  # Use sliding window for large images
        patch_size=128
    )
    # Pass the numpy array directly (preprocess_image will handle it)
    image_array = image_data.get_fdata() if hasattr(image_data, 'get_fdata') else image_data
    result1 = model.predict_comprehensive_with_preprocessing(
        image_array,
        affine=affine,
        apply_softmax=True,
        use_tta=False,
        use_sliding_window=True,  # Use sliding window for large images
        patch_size=128
    )
    # After inverse transform, output should match original spatial dimensions
    output_shape = result1['prediction'].shape
    passed1 = verify_dimensionality(original_shape, output_shape, "Basic inference")
    results['basic'] = {'result': result1, 'passed': passed1, 'config': config1}
    all_passed = all_passed and passed1
    
    # Save outputs
    saved_files1 = model.save_prediction(
        result1,
        output_dir / "test1_basic",
        base_filename="prediction",
        affine=affine
    )
    print(f"  Saved files: {list(saved_files1.keys())}")
    
    # Configuration 2: Sliding window only
    print("\n" + "="*60)
    print("Test 2: Sliding window inference (no TTA)")
    print("="*60)
    config2 = InferenceConfig(
        apply_softmax=True,
        use_tta=False,
        use_sliding_window=True,
        patch_size=128
    )
    result2 = model.predict_comprehensive_with_preprocessing(
        image_array,
        affine=affine,
        apply_softmax=True,
        use_tta=False,
        use_sliding_window=True,
        patch_size=128
    )
    output_shape2 = result2['prediction'].shape
    passed2 = verify_dimensionality(original_shape, output_shape2, "Sliding window")
    results['sliding_window'] = {'result': result2, 'passed': passed2, 'config': config2}
    all_passed = all_passed and passed2
    
    # Save outputs
    saved_files2 = model.save_prediction(
        result2,
        output_dir / "test2_sliding_window",
        base_filename="prediction",
        affine=affine
    )
    print(f"  Saved files: {list(saved_files2.keys())}")
    
    # Configuration 3: TTA only (if image fits in memory)
    print("\n" + "="*60)
    print("Test 3: TTA inference (no sliding window)")
    print("="*60)
    # Check if we need sliding window
    preprocessed_img, _ = model.preprocess_image(image_array, affine, is_ct=False, device=model.device)
    if preprocessed_img.dim() == 4:
        preprocessed_img = preprocessed_img.unsqueeze(0)
    
    needs_sw = model._needs_sliding_window(preprocessed_img, 128)
    if not needs_sw:
        config3 = InferenceConfig(
            apply_softmax=True,
            use_tta=True,
            use_sliding_window=False,
            patch_size=128
        )
        result3 = model.predict_comprehensive_with_preprocessing(
            image_array,
            affine=affine,
            apply_softmax=True,
            use_tta=True,
            use_sliding_window=False,
            patch_size=128
        )
        output_shape3 = result3['prediction'].shape
        passed3 = verify_dimensionality(original_shape, output_shape3, "TTA only")
        results['tta'] = {'result': result3, 'passed': passed3, 'config': config3}
        all_passed = all_passed and passed3
        
        # Save outputs
        saved_files3 = model.save_prediction(
            result3,
            output_dir / "test3_tta",
            base_filename="prediction",
            affine=affine
        )
        print(f"  Saved files: {list(saved_files3.keys())}")
    else:
        print("  Skipping TTA-only test (image requires sliding window)")
        results['tta'] = {'result': None, 'passed': None, 'config': None}
    
    # Configuration 4: Sliding window + TTA
    print("\n" + "="*60)
    print("Test 4: Sliding window + TTA")
    print("="*60)
    config4 = InferenceConfig(
        apply_softmax=True,
        use_tta=True,
        use_sliding_window=True,
        patch_size=128
    )
    result4 = model.predict_comprehensive_with_preprocessing(
        image_array,
        affine=affine,
        apply_softmax=True,
        use_tta=True,
        use_sliding_window=True,
        patch_size=128
    )
    output_shape4 = result4['prediction'].shape
    passed4 = verify_dimensionality(original_shape, output_shape4, "Sliding window + TTA")
    results['sliding_window_tta'] = {'result': result4, 'passed': passed4, 'config': config4}
    all_passed = all_passed and passed4
    
    # Save outputs
    saved_files4 = model.save_prediction(
        result4,
        output_dir / "test4_sliding_window_tta",
        base_filename="prediction",
        affine=affine
    )
    print(f"  Saved files: {list(saved_files4.keys())}")
    
    return results, all_passed


def main():
    """Main test function."""
    print("="*60)
    print("SynthStroke Model - End-to-End Test")
    print("Testing save_prediction and dimensionality preservation")
    print("="*60)
    
    # Setup paths
    test_data_dir = Path("test_data")
    test_image_path = test_data_dir / "mprage_img.nii"
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Check if test image exists
    if not test_image_path.exists():
        print(f"Error: Test image not found at {test_image_path}")
        return
    
    # Load test image
    print(f"\nLoading test image: {test_image_path}")
    image_nib = load_test_image(str(test_image_path))
    image_data = image_nib.get_fdata()
    affine = image_nib.affine
    print(f"  Image shape: {image_data.shape}")
    print(f"  Image dtype: {image_data.dtype}")
    print(f"  Image range: [{image_data.min():.2f}, {image_data.max():.2f}]")
    
    # Store original shape for verification
    original_shape = image_data.shape
    
    # Try to load a model from checkpoint
    checkpoint_dir = Path("checkpoints_for_hf")
    model_paths = {
        'baseline': checkpoint_dir / "baseline.safetensors",
        'synth': checkpoint_dir / "synth.safetensors",
    }
    
    model = None
    model_name = None
    
    # Try to load baseline first, then synth
    for name, path in model_paths.items():
        if path.exists():
            print(f"\nLoading model: {name} from {path}")
            try:
                model = SynthStrokeModel.from_checkpoint(str(path))
                model_name = name
                model.eval()
                if torch.cuda.is_available():
                    model = model.cuda()
                    print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
                else:
                    print("  Using CPU")
                break
            except Exception as e:
                print(f"  Failed to load {name}: {e}")
                continue
    
    if model is None:
        print("\nWarning: No model checkpoint found. Creating a dummy model for testing.")
        print("Note: Predictions will be random/meaningless, but dimensionality will be tested.")
        config = ModelConfig(out_channels=2, model_type="baseline")
        model = SynthStrokeModel(config=config)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        model_name = "dummy"
    
    print(f"\nModel info:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Run tests - pass the nibabel image object directly
    print("\n" + "="*60)
    print("Running inference tests...")
    print("="*60)
    
    results, all_passed = test_inference_configurations(
        model, image_nib, affine, output_dir, original_shape
    )
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for test_name, test_result in results.items():
        if test_result['passed'] is not None:
            status = "✓ PASSED" if test_result['passed'] else "✗ FAILED"
            print(f"{test_name:25s}: {status}")
        else:
            print(f"{test_name:25s}: SKIPPED")
    
    if all_passed:
        print("\n✓ All dimensionality tests PASSED!")
    else:
        print("\n✗ Some dimensionality tests FAILED!")
    
    print(f"\nOutput files saved to: {output_dir}")
    print("\nTest completed!")


if __name__ == "__main__":
    main()

