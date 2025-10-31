#!/usr/bin/env python3
"""
Block Structure Comparison
===========================

Compare different graph coloring strategies for block Gibbs sampling.
This demonstrates WHY block parallelization matters for TSU hardware!

We'll compare:
1. 2-coloring (checkerboard) - Maximum parallelism for 2D lattice
2. 4-coloring - Suboptimal but still valid
3. Sequential (no parallelism) - Baseline
"""

import jax
import jax.numpy as jnp
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
import matplotlib.pyplot as plt
import numpy as np
import time


def create_2d_lattice(grid_size):
    """Create 2D lattice structure."""
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


def strategy_2_coloring(nodes, grid_size):
    """Optimal 2-coloring (checkerboard pattern)."""
    even_indices = [i * grid_size + j for i in range(grid_size) 
                   for j in range(grid_size) if (i + j) % 2 == 0]
    odd_indices = [i * grid_size + j for i in range(grid_size) 
                  for j in range(grid_size) if (i + j) % 2 == 1]
    
    return [
        Block([nodes[i] for i in even_indices]),
        Block([nodes[i] for i in odd_indices])
    ]


def strategy_4_coloring(nodes, grid_size):
    """Suboptimal 4-coloring."""
    # Divide into 4 quadrants
    colors = [[] for _ in range(4)]
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            color = (i % 2) * 2 + (j % 2)
            colors[color].append(idx)
    
    return [Block([nodes[i] for i in color]) for color in colors]


def strategy_sequential(nodes, grid_size):
    """Sequential (one node at a time) - no parallelism."""
    # Each node is its own block
    return [Block([node]) for node in nodes[:grid_size]]  # Just first row for demo


def visualize_blocking_strategy(grid_size, strategy_name, blocks):
    """Visualize which nodes are in which block."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create color map for blocks
    colors = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261']
    
    grid = np.zeros((grid_size, grid_size))
    
    for block_idx, block in enumerate(blocks):
        for node in block.nodes:
            # Find position of this node
            for i in range(grid_size):
                for j in range(grid_size):
                    idx = i * grid_size + j
                    # This is a bit hacky but works for visualization
                    if str(node) == str(SpinNode()):  # Just mark by index
                        pass
            # Simplified: just color by block pattern
            pass
    
    # For now, just show the pattern conceptually
    if strategy_name == "2-Coloring":
        for i in range(grid_size):
            for j in range(grid_size):
                grid[i, j] = (i + j) % 2
    elif strategy_name == "4-Coloring":
        for i in range(grid_size):
            for j in range(grid_size):
                grid[i, j] = (i % 2) * 2 + (j % 2)
    elif strategy_name == "Sequential":
        grid[:1, :] = 0  # First row
        grid[1:, :] = 1  # Rest different
    
    im = ax.imshow(grid, cmap='tab10', interpolation='nearest')
    ax.set_title(f'{strategy_name}\n{len(blocks)} parallel blocks', 
                fontsize=16, weight='bold', pad=20)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add grid lines
    for i in range(grid_size + 1):
        ax.axhline(i - 0.5, color='white', linewidth=2)
        ax.axvline(i - 0.5, color='white', linewidth=2)
    
    return fig


def benchmark_strategy(grid_size, strategy_name, blocks, nodes, edges, n_samples=1000):
    """Benchmark a blocking strategy."""
    
    n_total = grid_size * grid_size
    
    biases = jnp.zeros((n_total,))
    weights = jnp.ones((len(edges),)) * 1.0
    beta = jnp.array(1.0)
    
    model = IsingEBM(nodes, edges, biases, weights, beta)
    program = IsingSamplingProgram(model, blocks, clamped_blocks=[])
    
    key = jax.random.key(42)
    k_init, k_samp = jax.random.split(key, 2)
    # Use hinton_init like working examples
    init_state = hinton_init(k_init, model, blocks, ())
    
    schedule = SamplingSchedule(
        n_warmup=50,
        n_samples=n_samples,
        steps_per_sample=1
    )
    
    # Warmup JIT compilation
    _ = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
    
    # Actual timing
    start = time.time()
    samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
    jax.block_until_ready(samples)
    elapsed = time.time() - start
    
    spin_samples = samples[0]
    magnetization = jnp.mean(jnp.abs(jnp.mean(spin_samples, axis=1)))
    
    return {
        'time': elapsed,
        'samples_per_sec': n_samples / elapsed,
        'magnetization': float(magnetization)
    }


def main():
    """Compare different blocking strategies."""
    
    GRID_SIZE = 16
    N_SAMPLES = 1000
    
    print("=" * 70)
    print("BLOCK STRUCTURE COMPARISON")
    print("=" * 70)
    print(f"Grid Size: {GRID_SIZE}x{GRID_SIZE} ({GRID_SIZE*GRID_SIZE} spins)")
    print(f"Samples: {N_SAMPLES}")
    print("=" * 70)
    
    nodes, edges = create_2d_lattice(GRID_SIZE)
    
    strategies = {
        "2-Coloring (Checkerboard)": strategy_2_coloring(nodes, GRID_SIZE),
        "4-Coloring": strategy_4_coloring(nodes, GRID_SIZE),
    }
    
    results = {}
    
    print("\n🔬 Running Benchmarks...\n")
    
    for name, blocks in strategies.items():
        print(f"{name}:")
        print(f"  • Number of blocks: {len(blocks)}")
        print(f"  • Nodes per block: {[len(b.nodes) for b in blocks]}")
        print(f"  • Benchmarking... ", end="", flush=True)
        
        result = benchmark_strategy(GRID_SIZE, name, blocks, nodes, edges, N_SAMPLES)
        results[name] = result
        
        print(f"✓")
        print(f"  • Time: {result['time']:.3f} sec")
        print(f"  • Throughput: {result['samples_per_sec']:.1f} samples/sec")
        print(f"  • Magnetization: {result['magnetization']:.3f}")
        print()
    
    # Analysis
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    baseline = results["2-Coloring (Checkerboard)"]
    
    print(f"\n📊 Performance Comparison (vs 2-Coloring baseline):\n")
    
    for name, result in results.items():
        speedup = baseline['samples_per_sec'] / result['samples_per_sec']
        print(f"{name}:")
        print(f"  • Relative speed: {1/speedup:.2f}x")
        print(f"  • Blocks: {len(strategies[name])}")
        print(f"  • Parallelism: {len(strategies[name])} parallel updates per step")
        print()
    
    print("=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    
    print("""
🎯 Why 2-Coloring is Optimal for 2D Ising:
  • Maximum parallelism: Half the spins update simultaneously
  • No conflicts: No adjacent spins in same block
  • Perfectly suited for TSU hardware architecture
  
🔧 Hardware Implications:
  • TSU can process entire blocks in parallel circuits
  • 2-coloring = 50% of spins active each timestep
  • 4-coloring = 25% of spins active (wasted capacity)
  
⚡ Extropic's Advantage:
  • GPU still serializes internally (SIMD parallelism)
  • TSU has true parallel sampling circuits
  • Block structure directly maps to hardware topology
    """)
    
    # Create visualization
    print("\n📊 Generating visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (name, blocks) in enumerate(strategies.items()):
        ax = axes[idx]
        
        if name == "2-Coloring (Checkerboard)":
            grid = np.zeros((GRID_SIZE, GRID_SIZE))
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    grid[i, j] = (i + j) % 2
        else:  # 4-coloring
            grid = np.zeros((GRID_SIZE, GRID_SIZE))
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    grid[i, j] = (i % 2) * 2 + (j % 2)
        
        im = ax.imshow(grid, cmap='tab10', interpolation='nearest')
        ax.set_title(f'{name}\n{len(blocks)} parallel blocks\n'
                    f'{results[name]["samples_per_sec"]:.0f} samples/sec',
                    fontsize=12, weight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Grid lines
        for i in range(GRID_SIZE + 1):
            ax.axhline(i - 0.5, color='white', linewidth=1, alpha=0.5)
            ax.axvline(i - 0.5, color='white', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    filename = f'blocking_comparison_{GRID_SIZE}x{GRID_SIZE}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {filename}")
    
    plt.show()
    
    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()
