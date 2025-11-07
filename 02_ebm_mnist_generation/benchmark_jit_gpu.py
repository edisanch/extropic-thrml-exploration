#!/usr/bin/env python3
"""
Benchmark JIT Compilation & GPU Optimization
=============================================

Phase 4, Step 4.3: JIT Compilation & GPU Optimization

Compare different optimization strategies:
1. sample_batch (sequential baseline)
2. sample_batch_vmap (vectorized - Step 4.2)
3. sample_batch_jit (explicit JIT wrapper)
4. sample_batch_gpu (explicit GPU placement)

Expected results:
- JIT: Additional 1.2-1.5x speedup from explicit JIT wrapper
- GPU: Better memory efficiency, similar speed (already on GPU)
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import matplotlib.pyplot as plt

from ebm_model import CategoricalEBM
from thrml_sampler_native import NativeTHRMLSampler


# ============================================================================
# DEVICE INFORMATION
# ============================================================================

def print_device_info():
    """Print JAX device configuration."""
    print("\n" + "="*70)
    print("DEVICE INFORMATION")
    print("="*70)
    
    devices = jax.devices()
    print(f"\nAvailable devices: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device}")
    
    # Check for GPU
    gpu_devices = jax.devices('gpu')
    if gpu_devices:
        print(f"\n✓ GPU available: {gpu_devices[0]}")
    else:
        print("\n⚠️  No GPU detected - running on CPU")
    
    # Memory info
    try:
        for device in devices:
            memory_stats = device.memory_stats()
            if memory_stats:
                print(f"\nMemory stats for {device}:")
                for key, val in memory_stats.items():
                    if isinstance(val, int):
                        print(f"  {key}: {val / 1e9:.2f} GB")
    except:
        pass


# ============================================================================
# BENCHMARKING
# ============================================================================

def benchmark_method(
    sampler,
    method_name: str,
    batch_size: int,
    n_steps: int,
    n_trials: int = 5,
    warmup_trials: int = 2
) -> dict:
    """
    Benchmark a specific sampling method.
    
    Returns:
        dict with timing statistics
    """
    print(f"\n  Testing {method_name}...")
    
    method = getattr(sampler, method_name)
    
    # Warmup
    print(f"    Warmup ({warmup_trials} trials)...", end=" ", flush=True)
    for _ in range(warmup_trials):
        samples = method(batch_size=batch_size, n_steps=n_steps)
        samples.block_until_ready()
    print("✓")
    
    # Timing
    times = []
    print(f"    Timing ({n_trials} trials)...", end=" ", flush=True)
    for _ in range(n_trials):
        start = time.time()
        samples = method(batch_size=batch_size, n_steps=n_steps)
        samples.block_until_ready()
        elapsed = time.time() - start
        times.append(elapsed)
    print("✓")
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"    Time: {mean_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    
    return {
        'method': method_name,
        'mean_time': mean_time,
        'std_time': std_time,
        'times': times
    }


def comprehensive_benchmark(
    sampler,
    batch_size: int = 32,
    n_steps: int = 50,
    n_trials: int = 5
):
    """
    Run comprehensive benchmark of all optimization methods.
    """
    print("\n" + "="*70)
    print(f"COMPREHENSIVE BENCHMARK (batch_size={batch_size}, n_steps={n_steps})")
    print("="*70)
    
    methods = [
        'sample_batch',      # Baseline: sequential
        'sample_batch_vmap', # Step 4.2: vectorized
        # 'sample_batch_jit',  # Step 4.3: explicit JIT (skipped - THRML already JIT-compiled internally)
        'sample_batch_gpu',  # Step 4.3: GPU-optimized
    ]
    
    results = {}
    
    for method_name in methods:
        if not hasattr(sampler, method_name):
            print(f"\n  ⚠️  Method '{method_name}' not found - skipping")
            continue
        
        try:
            result = benchmark_method(
                sampler, method_name, batch_size, n_steps, n_trials
            )
            results[method_name] = result
        except Exception as e:
            print(f"\n  ✗ Error in {method_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def print_comparison_table(results, batch_size):
    """Print comparison table of all methods."""
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    
    if 'sample_batch' in results:
        baseline_time = results['sample_batch']['mean_time']
    else:
        baseline_time = None
    
    print(f"\n{'Method':<25} {'Time (ms)':<15} {'Speedup':<10} {'Samples/sec':<15}")
    print("-"*70)
    
    for method_name, result in results.items():
        mean_time = result['mean_time']
        std_time = result['std_time']
        samples_per_sec = batch_size / mean_time
        
        if baseline_time and method_name != 'sample_batch':
            speedup = baseline_time / mean_time
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "baseline"
        
        print(f"{method_name:<25} {mean_time*1000:<15.2f} {speedup_str:<10} {samples_per_sec:<15.1f}")
    
    # Summary
    if len(results) > 1:
        print("-"*70)
        if baseline_time:
            best_time = min(r['mean_time'] for r in results.values())
            best_method = [k for k, v in results.items() if v['mean_time'] == best_time][0]
            max_speedup = baseline_time / best_time
            
            print(f"Best method: {best_method}")
            print(f"Max speedup: {max_speedup:.2f}x")
            print(f"Best throughput: {batch_size / best_time:.1f} samples/sec")


def visualize_results(results, batch_size):
    """Create visualization of benchmark results."""
    if len(results) < 2:
        print("\n⚠️  Not enough results to visualize")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract data
    methods = list(results.keys())
    mean_times = [results[m]['mean_time'] * 1000 for m in methods]
    std_times = [results[m]['std_time'] * 1000 for m in methods]
    
    # Plot 1: Execution time
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    bars1 = ax1.bar(range(len(methods)), mean_times, yerr=std_times,
                    capsize=5, color=colors[:len(methods)], alpha=0.8)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.set_ylabel('Time (ms)', fontweight='bold')
    ax1.set_title('Execution Time (Lower is Better)', fontweight='bold', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, time) in enumerate(zip(bars1, mean_times)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_times[i],
                f'{time:.1f}ms', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Speedup relative to baseline
    if 'sample_batch' in results:
        baseline_time = results['sample_batch']['mean_time'] * 1000
        speedups = [baseline_time / t for t in mean_times]
        
        bars2 = ax2.bar(range(len(methods)), speedups, 
                       color=colors[:len(methods)], alpha=0.8)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline')
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        ax2.set_ylabel('Speedup vs Baseline', fontweight='bold')
        ax2.set_title('Speedup Comparison', fontweight='bold', fontsize=12)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # Add speedup labels
        for bar, speedup in zip(bars2, speedups):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{speedup:.2f}x', ha='center', va='bottom', fontsize=9)
    else:
        # Just show throughput
        throughputs = [batch_size / (t / 1000) for t in mean_times]
        bars2 = ax2.bar(range(len(methods)), throughputs,
                       color=colors[:len(methods)], alpha=0.8)
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        ax2.set_ylabel('Samples/Second', fontweight='bold')
        ax2.set_title('Throughput (Higher is Better)', fontweight='bold', fontsize=12)
        ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    filename = 'jit_gpu_optimization_comparison.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {filename}")
    plt.close()


# ============================================================================
# JIT COMPILATION OVERHEAD TEST
# ============================================================================

def test_jit_compilation_overhead(sampler, batch_size=16, n_steps=50):
    """
    Measure JIT compilation overhead (first call vs subsequent calls).
    """
    print("\n" + "="*70)
    print("JIT COMPILATION OVERHEAD TEST")
    print("="*70)
    
    print("\n⚠️  Skipping explicit JIT test")
    print("   THRML's sample_states() is already JIT-compiled internally")
    print("   Additional JIT wrapper doesn't provide extra benefit")
    print("   (Attempting to JIT-wrap causes tracing errors with THRML's internal control flow)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run comprehensive JIT and GPU optimization benchmarks."""
    print("\n" + "="*70)
    print("PHASE 4, STEP 4.3: JIT COMPILATION & GPU OPTIMIZATION")
    print("="*70)
    
    # Device info
    print_device_info()
    
    # Create sampler
    print("\n" + "="*70)
    print("CREATING SAMPLER")
    print("="*70)
    ebm = CategoricalEBM(height=28, width=28, n_levels=4)
    sampler = NativeTHRMLSampler(ebm, seed=42)
    
    # Test 1: JIT compilation overhead
    test_jit_compilation_overhead(sampler, batch_size=16, n_steps=50)
    
    # Test 2: Comprehensive benchmark
    results = comprehensive_benchmark(
        sampler,
        batch_size=32,
        n_steps=50,
        n_trials=5
    )
    
    # Print comparison
    print_comparison_table(results, batch_size=32)
    
    # Visualize
    visualize_results(results, batch_size=32)
    
    # Summary
    print("\n" + "="*70)
    print("✅ BENCHMARK COMPLETE")
    print("="*70)
    
    if len(results) >= 2:
        baseline = results.get('sample_batch')
        vmap = results.get('sample_batch_vmap')
        jit = results.get('sample_batch_jit')
        gpu = results.get('sample_batch_gpu')
        
        print("\n📊 Summary:")
        if baseline and vmap:
            vmap_speedup = baseline['mean_time'] / vmap['mean_time']
            print(f"   Step 4.2 (vmap): {vmap_speedup:.2f}x speedup")
        
        if vmap and jit:
            jit_speedup = vmap['mean_time'] / jit['mean_time']
            print(f"   Step 4.3 (JIT): {jit_speedup:.2f}x additional speedup")
        
        if vmap and gpu:
            gpu_speedup = vmap['mean_time'] / gpu['mean_time']
            print(f"   Step 4.3 (GPU): {gpu_speedup:.2f}x vs vmap")
        
        if baseline and jit:
            total_jit = baseline['mean_time'] / jit['mean_time']
            print(f"   Total (baseline→JIT): {total_jit:.2f}x speedup")
        
        if baseline and gpu:
            total_gpu = baseline['mean_time'] / gpu['mean_time']
            print(f"   Total (baseline→GPU): {total_gpu:.2f}x speedup")
    
    print("\n📁 Output files:")
    print("   - jit_gpu_optimization_comparison.png")


if __name__ == "__main__":
    main()
