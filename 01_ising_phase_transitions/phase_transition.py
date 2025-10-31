#!/usr/bin/env python3
"""
Phase Transition Analysis
=========================

Sweep temperature and observe the magnetic phase transition!
This shows the critical behavior that makes the Ising model famous.

At the critical temperature Tc ≈ 2.27, the system transitions from
disordered (high T) to ordered (low T) states.
"""

import jax
import jax.numpy as jnp
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
import matplotlib.pyplot as plt
import numpy as np


def measure_magnetization(grid_size, coupling, temperature, n_samples=500):
    """Measure average magnetization at a given temperature."""
    
    # Create 2D lattice
    n_total = grid_size * grid_size
    nodes = [SpinNode() for _ in range(n_total)]
    
    # 2D lattice edges with periodic boundaries
    edges = []
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            right = i * grid_size + ((j + 1) % grid_size)
            edges.append((nodes[idx], nodes[right]))
            down = ((i + 1) % grid_size) * grid_size + j
            edges.append((nodes[idx], nodes[down]))
    
    biases = jnp.zeros((n_total,))
    weights = jnp.ones((len(edges),)) * coupling
    beta = 1.0 / max(temperature, 0.01)
    
    model = IsingEBM(nodes, edges, biases, weights, jnp.array(beta))
    
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
    
    # Sample
    key = jax.random.key(42)
    k_init, k_samp = jax.random.split(key, 2)
    init_state = hinton_init(k_init, model, free_blocks, ())
    
    schedule = SamplingSchedule(
        n_warmup=200,  # Longer warmup for equilibration
        n_samples=n_samples,
        steps_per_sample=2
    )
    
    samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
    spin_samples = samples[0]
    
    # Calculate magnetization per sample
    magnetization_per_sample = jnp.mean(spin_samples, axis=1)
    
    # Return statistics
    mean_mag = float(jnp.mean(magnetization_per_sample))
    abs_mag = float(jnp.mean(jnp.abs(magnetization_per_sample)))
    std_mag = float(jnp.std(magnetization_per_sample))
    
    return {
        'mean_magnetization': mean_mag,
        'abs_magnetization': abs_mag,
        'std_magnetization': std_mag,
        'samples': np.array(magnetization_per_sample)
    }


def sweep_temperature(grid_size=10, coupling=1.0, temp_range=(0.5, 4.0), n_temps=20):
    """Sweep through temperatures and measure magnetization."""
    
    temperatures = np.linspace(temp_range[0], temp_range[1], n_temps)
    
    print("=" * 60)
    print("PHASE TRANSITION ANALYSIS")
    print("=" * 60)
    print(f"Grid Size: {grid_size}x{grid_size}")
    print(f"Coupling J: {coupling}")
    print(f"Temperature Range: {temp_range[0]:.2f} to {temp_range[1]:.2f}")
    print(f"Number of Points: {n_temps}")
    print("=" * 60)
    
    results = {
        'temperatures': [],
        'mean_magnetization': [],
        'abs_magnetization': [],
        'std_magnetization': []
    }
    
    for i, temp in enumerate(temperatures):
        print(f"\nPoint {i+1}/{n_temps}: T = {temp:.3f} ... ", end="", flush=True)
        
        result = measure_magnetization(grid_size, coupling, temp, n_samples=300)
        
        results['temperatures'].append(temp)
        results['mean_magnetization'].append(result['mean_magnetization'])
        results['abs_magnetization'].append(result['abs_magnetization'])
        results['std_magnetization'].append(result['std_magnetization'])
        
        print(f"|m| = {result['abs_magnetization']:.3f}, std = {result['std_magnetization']:.3f}")
    
    return results


def plot_phase_transition(results, grid_size, coupling):
    """Create publication-quality phase transition plot."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    temps = results['temperatures']
    
    # Plot 1: Absolute magnetization (order parameter)
    ax1.plot(temps, results['abs_magnetization'], 'o-', 
            linewidth=2, markersize=8, color='#E63946', label='|⟨m⟩|')
    ax1.axvline(2.27, color='gray', linestyle='--', alpha=0.5, 
               label=f'Tc ≈ 2.27 (infinite lattice)')
    
    ax1.set_xlabel('Temperature (T)', fontsize=14, weight='bold')
    ax1.set_ylabel('|Magnetization|', fontsize=14, weight='bold')
    ax1.set_title(f'Ising Model Phase Transition ({grid_size}×{grid_size} lattice, J={coupling})', 
                 fontsize=16, weight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.set_ylim(-0.05, 1.05)
    
    # Add annotations
    ax1.annotate('Ordered\n(Ferromagnetic)', xy=(0.8, 0.9), fontsize=12,
                ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax1.annotate('Disordered\n(Paramagnetic)', xy=(3.5, 0.1), fontsize=12,
                ha='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    # Plot 2: Standard deviation (fluctuations)
    ax2.plot(temps, results['std_magnetization'], 's-', 
            linewidth=2, markersize=8, color='#457B9D', label='σ(m)')
    ax2.axvline(2.27, color='gray', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Temperature (T)', fontsize=14, weight='bold')
    ax2.set_ylabel('Magnetization Fluctuations (σ)', fontsize=14, weight='bold')
    ax2.set_title('Critical Fluctuations Peak at Tc', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    
    # Add annotation for critical point
    max_std_idx = np.argmax(results['std_magnetization'])
    max_std_temp = temps[max_std_idx]
    max_std_val = results['std_magnetization'][max_std_idx]
    ax2.annotate(f'Peak fluctuations\nT ≈ {max_std_temp:.2f}', 
                xy=(max_std_temp, max_std_val),
                xytext=(max_std_temp + 0.5, max_std_val + 0.05),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', weight='bold')
    
    plt.tight_layout()
    
    # Save figure
    filename = f'phase_transition_{grid_size}x{grid_size}_J{coupling}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to: {filename}")
    
    plt.show()
    
    return filename


def main():
    """Run complete phase transition analysis."""
    
    # Parameters
    GRID_SIZE = 8  # Smaller for faster demo
    COUPLING = 1.0
    TEMP_MIN = 0.5
    TEMP_MAX = 4.0
    N_POINTS = 15  # Fewer points for faster run
    
    print("\n🌡️  ISING MODEL PHASE TRANSITION EXPLORER\n")
    
    # Run temperature sweep
    results = sweep_temperature(
        grid_size=GRID_SIZE,
        coupling=COUPLING,
        temp_range=(TEMP_MIN, TEMP_MAX),
        n_temps=N_POINTS
    )
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    # Find critical temperature (where fluctuations peak)
    max_std_idx = np.argmax(results['std_magnetization'])
    estimated_tc = results['temperatures'][max_std_idx]
    
    print(f"\n📊 Key Results:")
    print(f"  • Estimated Tc: {estimated_tc:.2f}")
    print(f"  • Theory Tc (∞ lattice): 2.27")
    print(f"  • Finite-size effects: {abs(estimated_tc - 2.27):.2f}")
    print(f"\n  • Low T magnetization: {results['abs_magnetization'][0]:.3f}")
    print(f"  • High T magnetization: {results['abs_magnetization'][-1]:.3f}")
    
    print("\n" + "=" * 60)
    print("GENERATING PLOTS...")
    print("=" * 60)
    
    # Create visualization
    plot_phase_transition(results, GRID_SIZE, COUPLING)
    
    print("\n✓ Analysis complete!")
    print("\n💡 What you're seeing:")
    print("  • Sharp transition from ordered (low T) to disordered (high T)")
    print("  • Fluctuations peak at critical temperature")
    print("  • Spontaneous magnetization appears below Tc")
    print("  • This is what makes statistical mechanics beautiful!")
    
    print("\n📖 Further exploration:")
    print("  • Increase grid size to see Tc approach 2.27")
    print("  • Try different coupling strengths J")
    print("  • Add external field h to break symmetry")
    print("  • Compare to 1D (no transition) or 3D (Tc ≈ 4.5)")


if __name__ == "__main__":
    main()
