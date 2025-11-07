#!/usr/bin/env python3
"""
Quick Test for Vectorized Batch Sampling
=========================================

Test that sample_batch_vmap() works correctly before running full benchmark.
"""

import jax
import numpy as np
from ebm_model import CategoricalEBM
from thrml_sampler_native import NativeTHRMLSampler

print("\n" + "="*70)
print("TESTING VECTORIZED BATCH SAMPLING")
print("="*70)

# Create model and sampler
print("\n📦 Creating sampler...")
ebm = CategoricalEBM(height=28, width=28, n_levels=4)
sampler = NativeTHRMLSampler(ebm, seed=42)

print(f"\nJAX backend: {jax.devices()}")

# Test 1: Basic functionality
print("\n" + "="*70)
print("TEST 1: Basic Functionality")
print("="*70)

batch_size = 4
n_steps = 10

print(f"\nSampling {batch_size} images with {n_steps} steps...")

try:
    samples = sampler.sample_batch_vmap(batch_size=batch_size, n_steps=n_steps)
    print(f"✓ Success! Shape: {samples.shape}")
    print(f"  Value range: [{samples.min()}, {samples.max()}]")
    assert samples.shape == (batch_size, 784), f"Expected shape (4, 784), got {samples.shape}"
    assert samples.min() >= 0 and samples.max() < 4, "Values out of range!"
    print("✓ All checks passed!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 2: Compare with sequential
print("\n" + "="*70)
print("TEST 2: Compare with Sequential")
print("="*70)

print("\nSampling with both methods...")
samples_seq = sampler.sample_batch(batch_size=batch_size, n_steps=n_steps)
samples_vmap = sampler.sample_batch_vmap(batch_size=batch_size, n_steps=n_steps)

print(f"Sequential shape: {samples_seq.shape}")
print(f"Vectorized shape: {samples_vmap.shape}")

assert samples_seq.shape == samples_vmap.shape, "Shapes don't match!"
print("✓ Shapes match!")

# Test 3: Different batch sizes
print("\n" + "="*70)
print("TEST 3: Different Batch Sizes")
print("="*70)

for batch_size in [1, 8, 16]:
    print(f"\n  Testing batch_size={batch_size}...")
    samples = sampler.sample_batch_vmap(batch_size=batch_size, n_steps=5)
    print(f"    Shape: {samples.shape}")
    assert samples.shape == (batch_size, 784)
    print(f"    ✓ Passed")

print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\n➡️  Ready to run full benchmark: python benchmark_batch_sampling.py")
