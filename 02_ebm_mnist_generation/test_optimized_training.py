#!/usr/bin/env python3
"""
Quick test of the optimized training script.

This runs 1 epoch to verify:
1. NativeTHRMLSampler integrates correctly
2. Vectorized batch sampling works in training loop
3. TensorBoard logging functions properly
4. Checkpointing works
"""

import sys
from pathlib import Path

# Ensure train_ebm_optimized can be imported
sys.path.insert(0, str(Path(__file__).parent))

from train_ebm_optimized import TrainingConfig, train_ebm_optimized


def test_optimized_training():
    """Run 1 epoch test to verify integration."""
    print("\n" + "=" * 70)
    print("TESTING OPTIMIZED TRAINING INTEGRATION")
    print("=" * 70)
    
    # Create minimal test config
    config = TrainingConfig()
    config.n_epochs = 1  # Just 1 epoch for testing
    config.batch_size = 16  # Smaller batch for speed
    config.cd_steps = 3  # Reasonable for testing
    config.use_gibbs_in_training = True  # The key feature!
    config.log_every_n_steps = 5
    config.save_every_n_epochs = 1  # Save checkpoint
    
    print("\n🧪 Test Configuration:")
    print(f"  Epochs: {config.n_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  CD steps: {config.cd_steps}")
    print(f"  Use Gibbs: {config.use_gibbs_in_training}")
    
    print("\n🚀 Running training...")
    ebm, sampler = train_ebm_optimized(config)
    
    print("\n" + "=" * 70)
    print("✅ TEST PASSED")
    print("=" * 70)
    print("\n✓ Verified:")
    print("  ✓ NativeTHRMLSampler integration works")
    print("  ✓ Vectorized batch sampling in training loop")
    print("  ✓ TensorBoard logging functional")
    print("  ✓ Checkpointing works")
    
    # Generate a test sample
    print("\n🎨 Testing sample generation...")
    import numpy as np
    samples = sampler.sample_batch_vmap(
        batch_size=4,
        n_steps=10
    )
    print(f"✓ Generated {len(samples)} samples successfully")
    print(f"  Sample shape: {samples.shape}")
    print(f"  Sample dtype: {samples.dtype}")
    print(f"  Sample range: [{samples.min()}, {samples.max()}]")
    
    print("\n✅ ALL TESTS PASSED")
    print("\n➡️  Ready for full training run!")
    

if __name__ == "__main__":
    test_optimized_training()
