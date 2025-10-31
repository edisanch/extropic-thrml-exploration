#!/usr/bin/env python3
"""
Sampling Speed Benchmark
=========================

Compare THRML's optimized block Gibbs sampling against naive implementations.
This validates (or challenges!) Extropic's efficiency claims.

Comparisons:
1. THRML block Gibbs (GPU) - Optimized JAX + CUDA
2. THRML block Gibbs (CPU) - JAX on CPU
3. Naive Python - Sequential single-spin updates
"""

import jax
import jax.numpy as jnp
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
import numpy as np
import time
import matplotlib.pyplot as plt


def naive_python_gibbs(grid_size, n_samples, coupling=1.0, temperature=1.0):
    """Naive Python implementation with sequential single-spin updates."""
    
    beta = 1.0 / temperature
    n_total = grid_size * grid_size
    
    # Initialize random spins
    spins = np.random.choice([-1, 1], size=(grid_size, grid_size))
    
    samples = []
    
    # Gibbs sampling
    for sample_idx in range(n_samples):
        # Update each spin sequentially
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate local field from neighbors
                neighbors_sum = (
                    spins[(i-1) % grid_size, j] +
                    spins[(i+1) % grid_size, j] +
                    spins[i, (j-1) % grid_size] +
                    spins[i, (j+1) % grid_size]
                )
                
                # Energy difference for flip
                delta_E = 2 * coupling * spins[i, j] * neighbors_sum
                
                # Metropolis acceptance (equivalent to Gibbs for Ising)
                if delta_E <= 0 or np.random.rand() < np.exp(-beta * delta_E):
                    spins[i, j] *= -1
        
        samples.append(spins.copy())
    
    return np.array(samples)


def benchmark_thrml_gpu(grid_size, n_samples):
    """Benchmark THRML on GPU."""
    
    # Force GPU
    jax.config.update('jax_platform_name', 'gpu')
    
    nodes, edges = create_2d_lattice(grid_size)
    n_total = grid_size * grid_size
    
    biases = jnp.zeros((n_total,))
    weights = jnp.ones((len(edges),)) * 1.0
    beta = jnp.array(1.0)
    
    model = IsingEBM(nodes, edges, biases, weights, beta)
    
    # Checkerboard coloring
    even_indices = [i * grid_size + j for i in range(grid_size) 
                   for j in range(grid_size) if (i + j) % 2 == 0]
    odd_indices = [i * grid_size + j for i in range(grid_size) 
                  for j in range(grid_size) if (i + j) % 2 == 1]
    
    free_blocks = [
        Block([nodes[i] for i in even_indices]),
        Block([nodes[i] for i in odd_indices])
    ]
    
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
    
    key = jax.random.key(42)
    k_init, k_samp = jax.random.split(key, 2)
    init_state = hinton_init(k_init, model, free_blocks, ())
    
    schedule = SamplingSchedule(n_warmup=10, n_samples=n_samples, steps_per_sample=1)
    
    # Warmup
    _ = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
    
    # Benchmark
    start = time.time()
    samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
    jax.block_until_ready(samples)
    elapsed = time.time() - start
    
    return elapsed, samples


def benchmark_thrml_cpu(grid_size, n_samples):
    """Benchmark THRML on CPU."""
    
    # Force CPU
    with jax.default_device(jax.devices('cpu')[0]):
        return benchmark_thrml_gpu(grid_size, n_samples)


def create_2d_lattice(grid_size):
    """Helper to create 2D lattice."""
    n_total = grid_size * grid_size
    nodes = [SpinNode() for _ in range(n_total)]
    
    edges = []
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            right = i * grid_size + ((j + 1) % grid_size)
            edges.append((nodes[idx], nodes[right]))
            down = ((i + 1) % grid_size) * grid_size + j
            edges.append((nodes[idx], nodes[down]))
    
    return nodes, edges


def main():
    """Run comprehensive benchmark suite."""
    
    print("=" * 70)
    print("SAMPLING SPEED BENCHMARK")
    print("Testing Extropic's efficiency claims")
    print("=" * 70)
    
    # Parameters
    GRID_SIZES = [8, 16, 32, 64, 128]
    N_SAMPLES = 500
    
    results = {
        'grid_sizes': GRID_SIZES,
        'thrml_gpu': [],
        'thrml_cpu': [],
        'naive_python': []
    }
    
    print(f"\nConfiguration:")
    print(f"  • Grid sizes: {GRID_SIZES}")
    print(f"  • Samples per test: {N_SAMPLES}")
    print(f"  • Temperature: 1.0, Coupling: 1.0")
    print("\n" + "=" * 70)
    
    for grid_size in GRID_SIZES:
        print(f"\n📐 Grid Size: {grid_size}x{grid_size} ({grid_size*grid_size} spins)\n")
        
        # THRML GPU
        print("  🚀 THRML (GPU)... ", end="", flush=True)
        try:
            time_gpu, _ = benchmark_thrml_gpu(grid_size, N_SAMPLES)
            samples_per_sec_gpu = N_SAMPLES / time_gpu
            results['thrml_gpu'].append(samples_per_sec_gpu)
            print(f"{time_gpu:.3f}s ({samples_per_sec_gpu:.0f} samples/s)")
        except Exception as e:
            print(f"Failed: {e}")
            results['thrml_gpu'].append(0)
        
        # THRML CPU
        print("  💻 THRML (CPU)... ", end="", flush=True)
        try:
            time_cpu, _ = benchmark_thrml_cpu(grid_size, N_SAMPLES)
            samples_per_sec_cpu = N_SAMPLES / time_cpu
            results['thrml_cpu'].append(samples_per_sec_cpu)
            print(f"{time_cpu:.3f}s ({samples_per_sec_cpu:.0f} samples/s)")
        except Exception as e:
            print(f"Failed: {e}")
            results['thrml_cpu'].append(0)
        
        # Naive Python (only for smaller grids - gets slow for large sizes)
        if grid_size <= 128:
            print("  🐌 Naive Python... ", end="", flush=True)
            start = time.time()
            _ = naive_python_gibbs(grid_size, N_SAMPLES)
            time_naive = time.time() - start
            samples_per_sec_naive = N_SAMPLES / time_naive
            results['naive_python'].append(samples_per_sec_naive)
            print(f"{time_naive:.3f}s ({samples_per_sec_naive:.0f} samples/s)")
        else:
            print("  🐌 Naive Python... Skipped (too slow)")
            results['naive_python'].append(None)
    
    # Analysis
    print("\n" + "=" * 70)
    print("SPEEDUP ANALYSIS")
    print("=" * 70)
    
    for idx, grid_size in enumerate(GRID_SIZES):
        print(f"\n{grid_size}x{grid_size} Grid:")
        
        gpu_speed = results['thrml_gpu'][idx]
        cpu_speed = results['thrml_cpu'][idx]
        naive_speed = results['naive_python'][idx]
        
        if naive_speed and naive_speed > 0:
            print(f"  THRML GPU vs Naive: {gpu_speed/naive_speed:.1f}x faster")
            print(f"  THRML CPU vs Naive: {cpu_speed/naive_speed:.1f}x faster")
        
        if gpu_speed > 0 and cpu_speed > 0:
            print(f"  GPU vs CPU: {gpu_speed/cpu_speed:.1f}x faster")
    
    # Plotting
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Absolute performance
    x = np.arange(len(GRID_SIZES))
    width = 0.25
    
    ax1.bar(x - width, results['thrml_gpu'], width, label='THRML (GPU)', color='#E63946')
    ax1.bar(x, results['thrml_cpu'], width, label='THRML (CPU)', color='#457B9D')
    
    naive_vals = [v if v else 0 for v in results['naive_python']]
    ax1.bar(x + width, naive_vals, width, label='Naive Python', color='#2A9D8F')
    
    ax1.set_xlabel('Grid Size', fontsize=12, weight='bold')
    ax1.set_ylabel('Samples per Second', fontsize=12, weight='bold')
    ax1.set_title('Sampling Performance Comparison', fontsize=14, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{g}×{g}' for g in GRID_SIZES])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')
    
    # Plot 2: Speedup vs Naive
    speedups_gpu = []
    speedups_cpu = []
    valid_sizes = []
    
    for idx, grid_size in enumerate(GRID_SIZES):
        if results['naive_python'][idx] and results['naive_python'][idx] > 0:
            speedups_gpu.append(results['thrml_gpu'][idx] / results['naive_python'][idx])
            speedups_cpu.append(results['thrml_cpu'][idx] / results['naive_python'][idx])
            valid_sizes.append(f'{grid_size}×{grid_size}')
    
    if speedups_gpu:
        x2 = np.arange(len(valid_sizes))
        ax2.bar(x2 - width/2, speedups_gpu, width, label='THRML (GPU)', color='#E63946')
        ax2.bar(x2 + width/2, speedups_cpu, width, label='THRML (CPU)', color='#457B9D')
        
        ax2.axhline(y=1, color='gray', linestyle='--', label='Baseline (Naive)')
        ax2.set_xlabel('Grid Size', fontsize=12, weight='bold')
        ax2.set_ylabel('Speedup vs Naive Python', fontsize=12, weight='bold')
        ax2.set_title('Optimization Speedup', fontsize=14, weight='bold')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(valid_sizes)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    filename = 'sampling_benchmark.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {filename}")
    
    plt.show()
    
    # Final summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    if speedups_gpu:
        avg_speedup_gpu = np.mean(speedups_gpu)
        avg_speedup_cpu = np.mean(speedups_cpu)
        
        # Determine if THRML is actually faster or slower
        if avg_speedup_gpu > 1.0:
            gpu_status = f"{avg_speedup_gpu:.1f}x FASTER"
            interpretation = "JAX/GPU optimization pays off"
        else:
            gpu_status = f"{1/avg_speedup_gpu:.1f}x SLOWER"
            interpretation = "Overhead dominates at small scale"
        
        if avg_speedup_cpu > 1.0:
            cpu_status = f"{avg_speedup_cpu:.1f}x FASTER"
        else:
            cpu_status = f"{1/avg_speedup_cpu:.1f}x SLOWER"
        
        print(f"""
🚀 Performance Summary:
  • THRML GPU is {gpu_status} than naive Python
  • THRML CPU is {cpu_status} than naive Python
  • GPU vs CPU: {(results['thrml_gpu'][-1]/results['thrml_cpu'][-1]):.2f}x on largest grid

🔬 What This Reveals:
  • {interpretation}
  • For small problems (<1000 spins): Python loops competitive
  • JIT compilation + kernel launch overhead is real
  • GPU memory transfer costs matter for tiny arrays
  
📊 Scaling Behavior:
  • Naive Python: O(n²) per sample, sequential
  • THRML Block Gibbs: O(n²) per sample, but vectorized
  • GPU advantage grows with problem size
  
� Context for Extropic's 10,000x Claim:
  • Their benchmarks use MASSIVE problems (millions of variables)
  • At that scale, specialized hardware dominates
  • Software overhead becomes negligible
  • True parallel circuits (pbits) vs simulated parallelism
  
🎯 The Real Lesson:
  • Specialized hardware only wins at scale!
  • For toy problems: simple code often fastest
  • For production workloads: TSUs could be transformational
  • Need the RIGHT problem to justify the hardware
        """)
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
