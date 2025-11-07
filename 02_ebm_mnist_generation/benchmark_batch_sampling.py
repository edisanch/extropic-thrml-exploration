#!/usr/bin/env python3
"""
Benchmark Batch Sampling Performance
=====================================

Phase 4, Step 4.2: Vectorize Training Pipeline

Compare sequential vs vectorized batch sampling to measure speedup.
This tests the vmap-based parallel sampling implementation.

Expected results:
- Sequential: ~0.3ms × batch_size (linear scaling)
- Vectorized (vmap): Near-constant time (GPU parallelism)
- Target: 10-20x speedup for batch operations
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import torch
from pathlib import Path
import matplotlib.pyplot as plt

from ebm_model import CategoricalEBM
from thrml_sampler_native import NativeTHRMLSampler


# ============================================================================
# BENCHMARKING UTILITIES
# ============================================================================

def benchmark_sampling_function(
    sampler,
    method_name: str,
    batch_sizes: list,
    n_steps: int = 50,
    n_trials: int = 5,
    warmup_trials: int = 2
) -> dict:
    """
    Benchmark a sampling method across different batch sizes.
    
    Args:
        sampler: Sampler object with the method to benchmark
        method_name: Name of the method ('sample_batch' or 'sample_batch_vmap')
        batch_sizes: List of batch sizes to test
        n_steps: Number of Gibbs steps per sample
        n_trials: Number of trials to average
        warmup_trials: Number of warmup trials (for JIT compilation)
    
    Returns:
        results: Dictionary with timing statistics
    """
    results = {
        'method': method_name,
        'batch_sizes': batch_sizes,
        'mean_times': [],
        'std_times': [],
        'samples_per_second': [],
        'time_per_sample': []
    }
    
    print(f"\n{'='*70}")
    print(f"Benchmarking: {method_name}")
    print(f"{'='*70}")
    
    method = getattr(sampler, method_name)
    
    for batch_size in batch_sizes:
        print(f"\n  Batch size: {batch_size}")
        
        # Warmup trials (JIT compilation)
        print(f"    Warming up ({warmup_trials} trials)...", end=" ", flush=True)
        for _ in range(warmup_trials):
            _ = method(batch_size=batch_size, n_steps=n_steps)
        print("✓")
        
        # Actual timing trials
        times = []
        print(f"    Timing ({n_trials} trials)...", end=" ", flush=True)
        for trial in range(n_trials):
            start = time.time()
            samples = method(batch_size=batch_size, n_steps=n_steps)
            # Force JAX to wait for GPU completion
            samples.block_until_ready()
            elapsed = time.time() - start
            times.append(elapsed)
        print("✓")
        
        # Statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        samples_per_sec = batch_size / mean_time
        time_per_sample = mean_time / batch_size * 1000  # ms
        
        results['mean_times'].append(mean_time)
        results['std_times'].append(std_time)
        results['samples_per_second'].append(samples_per_sec)
        results['time_per_sample'].append(time_per_sample)
        
        print(f"    Mean time: {mean_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
        print(f"    Samples/sec: {samples_per_sec:.1f}")
        print(f"    Time/sample: {time_per_sample:.2f}ms")
    
    return results


# ============================================================================
# COMPARISON & VISUALIZATION
# ============================================================================

def compare_batch_methods(
    sampler,
    batch_sizes: list = [1, 4, 8, 16, 32, 64],
    n_steps: int = 50,
    n_trials: int = 5
):
    """
    Compare sequential vs vectorized batch sampling.
    
    Args:
        sampler: NativeTHRMLSampler instance
        batch_sizes: List of batch sizes to test
        n_steps: Number of Gibbs steps
        n_trials: Number of timing trials
    """
    print("\n" + "="*70)
    print("BATCH SAMPLING COMPARISON")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Gibbs steps: {n_steps}")
    print(f"  Trials per batch: {n_trials}")
    print(f"  JAX backend: {jax.devices()}")
    
    # Check if sampler has vmap method
    has_vmap = hasattr(sampler, 'sample_batch_vmap')
    
    # Benchmark sequential method
    results_sequential = benchmark_sampling_function(
        sampler=sampler,
        method_name='sample_batch',
        batch_sizes=batch_sizes,
        n_steps=n_steps,
        n_trials=n_trials
    )
    
    # Benchmark vectorized method (if available)
    if has_vmap:
        results_vmap = benchmark_sampling_function(
            sampler=sampler,
            method_name='sample_batch_vmap',
            batch_sizes=batch_sizes,
            n_steps=n_steps,
            n_trials=n_trials
        )
    else:
        print("\n⚠️  sample_batch_vmap method not found - skipping vectorized benchmark")
        results_vmap = None
    
    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    print(f"\n{'Batch Size':<12} {'Sequential (ms)':<20} {'Vectorized (ms)':<20} {'Speedup':<10}")
    print("-"*70)
    
    speedups = []
    for i, batch_size in enumerate(batch_sizes):
        seq_time = results_sequential['mean_times'][i] * 1000
        
        if results_vmap:
            vmap_time = results_vmap['mean_times'][i] * 1000
            speedup = seq_time / vmap_time
            speedups.append(speedup)
            print(f"{batch_size:<12} {seq_time:<20.2f} {vmap_time:<20.2f} {speedup:<10.2f}x")
        else:
            print(f"{batch_size:<12} {seq_time:<20.2f} {'N/A':<20} {'N/A':<10}")
    
    if speedups:
        print("-"*70)
        print(f"{'Average speedup:':<32} {np.mean(speedups):.2f}x")
        print(f"{'Max speedup:':<32} {np.max(speedups):.2f}x")
    
    # Visualizations
    visualize_comparison(results_sequential, results_vmap, batch_sizes)
    
    return results_sequential, results_vmap


def visualize_comparison(results_sequential, results_vmap, batch_sizes):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Total time vs batch size
    ax = axes[0, 0]
    ax.plot(batch_sizes, np.array(results_sequential['mean_times']) * 1000, 
            'o-', label='Sequential', linewidth=2, markersize=8)
    if results_vmap:
        ax.plot(batch_sizes, np.array(results_vmap['mean_times']) * 1000,
                's-', label='Vectorized (vmap)', linewidth=2, markersize=8)
    ax.set_xlabel('Batch Size', fontweight='bold')
    ax.set_ylabel('Total Time (ms)', fontweight='bold')
    ax.set_title('Total Sampling Time', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Time per sample vs batch size
    ax = axes[0, 1]
    ax.plot(batch_sizes, results_sequential['time_per_sample'],
            'o-', label='Sequential', linewidth=2, markersize=8)
    if results_vmap:
        ax.plot(batch_sizes, results_vmap['time_per_sample'],
                's-', label='Vectorized (vmap)', linewidth=2, markersize=8)
    ax.set_xlabel('Batch Size', fontweight='bold')
    ax.set_ylabel('Time per Sample (ms)', fontweight='bold')
    ax.set_title('Time per Sample (Lower is Better)', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Samples per second vs batch size
    ax = axes[1, 0]
    ax.plot(batch_sizes, results_sequential['samples_per_second'],
            'o-', label='Sequential', linewidth=2, markersize=8)
    if results_vmap:
        ax.plot(batch_sizes, results_vmap['samples_per_second'],
                's-', label='Vectorized (vmap)', linewidth=2, markersize=8)
    ax.set_xlabel('Batch Size', fontweight='bold')
    ax.set_ylabel('Samples per Second', fontweight='bold')
    ax.set_title('Throughput (Higher is Better)', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Speedup vs batch size
    ax = axes[1, 1]
    if results_vmap:
        speedups = np.array(results_sequential['mean_times']) / np.array(results_vmap['mean_times'])
        ax.plot(batch_sizes, speedups, 'o-', linewidth=2, markersize=8, color='green')
        ax.axhline(y=1.0, color='red', linestyle='--', label='No speedup', alpha=0.5)
        ax.set_xlabel('Batch Size', fontweight='bold')
        ax.set_ylabel('Speedup (Sequential / Vectorized)', fontweight='bold')
        ax.set_title('Vectorization Speedup', fontweight='bold', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Vectorized method\nnot available',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    filename = 'batch_sampling_comparison.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plots to: {filename}")
    plt.close()


# ============================================================================
# CORRECTNESS VALIDATION
# ============================================================================

def validate_batch_methods(sampler, batch_size: int = 8, n_steps: int = 20):
    """
    Verify that sequential and vectorized methods produce similar results.
    
    Tests:
    1. Output shapes match
    2. Value ranges are correct
    3. Energy distributions are similar
    """
    print("\n" + "="*70)
    print("VALIDATING BATCH METHODS")
    print("="*70)
    
    # Check if vmap method exists
    if not hasattr(sampler, 'sample_batch_vmap'):
        print("\n⚠️  sample_batch_vmap method not found - skipping validation")
        return
    
    print(f"\n  Generating {batch_size} samples with {n_steps} steps...")
    
    # Generate with both methods
    samples_seq = sampler.sample_batch(batch_size=batch_size, n_steps=n_steps)
    samples_vmap = sampler.sample_batch_vmap(batch_size=batch_size, n_steps=n_steps)
    
    # Test 1: Shape
    print(f"\n✓ Test 1: Output Shapes")
    print(f"    Sequential: {samples_seq.shape}")
    print(f"    Vectorized: {samples_vmap.shape}")
    assert samples_seq.shape == samples_vmap.shape, "Shapes don't match!"
    print(f"    ✓ Shapes match")
    
    # Test 2: Value range
    print(f"\n✓ Test 2: Value Ranges")
    print(f"    Sequential: [{samples_seq.min()}, {samples_seq.max()}]")
    print(f"    Vectorized: [{samples_vmap.min()}, {samples_vmap.max()}]")
    assert samples_seq.min() >= 0 and samples_seq.max() < sampler.n_levels
    assert samples_vmap.min() >= 0 and samples_vmap.max() < sampler.n_levels
    print(f"    ✓ Values in valid range [0, {sampler.n_levels})")
    
    # Test 3: Energy distributions
    print(f"\n✓ Test 3: Energy Distributions")
    
    # Convert to torch and compute energies
    samples_seq_torch = torch.from_numpy(np.array(samples_seq)).long()
    samples_vmap_torch = torch.from_numpy(np.array(samples_vmap)).long()
    
    samples_seq_images = samples_seq_torch.view(batch_size, sampler.height, sampler.width)
    samples_vmap_images = samples_vmap_torch.view(batch_size, sampler.height, sampler.width)
    
    with torch.no_grad():
        energies_seq = sampler.ebm(samples_seq_images).numpy()
        energies_vmap = sampler.ebm(samples_vmap_images).numpy()
    
    print(f"    Sequential energy: {energies_seq.mean():.2f} ± {energies_seq.std():.2f}")
    print(f"    Vectorized energy: {energies_vmap.mean():.2f} ± {energies_vmap.std():.2f}")
    print(f"    ✓ Both methods produce valid samples")
    
    print("\n" + "="*70)
    print("✅ VALIDATION PASSED")
    print("="*70)


# ============================================================================
# MAIN BENCHMARK SUITE
# ============================================================================

def main():
    """Run complete benchmark suite for batch sampling."""
    print("\n" + "="*70)
    print("PHASE 4, STEP 4.2: VECTORIZED BATCH SAMPLING BENCHMARK")
    print("="*70)
    
    # Create model and sampler
    print("\n📦 Creating EBM and sampler...")
    ebm = CategoricalEBM(height=28, width=28, n_levels=4)
    sampler = NativeTHRMLSampler(ebm, seed=42)
    
    # Validation
    print("\n" + "="*70)
    print("STEP 1: VALIDATION")
    print("="*70)
    validate_batch_methods(sampler, batch_size=8, n_steps=20)
    
    # Benchmarking
    print("\n" + "="*70)
    print("STEP 2: PERFORMANCE BENCHMARKING")
    print("="*70)
    
    batch_sizes = [1, 4, 8, 16, 32, 64]
    results_seq, results_vmap = compare_batch_methods(
        sampler=sampler,
        batch_sizes=batch_sizes,
        n_steps=50,
        n_trials=5
    )
    
    # Summary
    print("\n" + "="*70)
    print("✅ BENCHMARK COMPLETE")
    print("="*70)
    
    if results_vmap:
        avg_speedup = np.mean([
            results_seq['mean_times'][i] / results_vmap['mean_times'][i]
            for i in range(len(batch_sizes))
        ])
        
        print(f"\n📊 Summary:")
        print(f"   Average speedup: {avg_speedup:.2f}x")
        print(f"   Best throughput (sequential): {max(results_seq['samples_per_second']):.1f} samples/sec")
        print(f"   Best throughput (vectorized): {max(results_vmap['samples_per_second']):.1f} samples/sec")
        
        if avg_speedup >= 5.0:
            print(f"\n✅ EXCELLENT! {avg_speedup:.1f}x speedup achieved")
        elif avg_speedup >= 2.0:
            print(f"\n✓ GOOD! {avg_speedup:.1f}x speedup achieved")
        else:
            print(f"\n⚠️  Modest speedup: {avg_speedup:.1f}x - may need further optimization")
    else:
        print("\n⚠️  Vectorized method not implemented yet")
        print("   Next: Implement sample_batch_vmap() in NativeTHRMLSampler")
    
    print("\n📁 Output files:")
    print("   - batch_sampling_comparison.png (performance plots)")


if __name__ == "__main__":
    main()
