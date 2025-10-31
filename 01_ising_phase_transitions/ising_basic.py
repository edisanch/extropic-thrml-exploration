#!/usr/bin/env python3
"""
Simple 1D Ising Chain Example
==============================

This is the simplest possible example of using THRML to sample from an Ising model.
It directly follows the quick example from the THRML README.

Run this first to verify your setup works!
"""

import jax
import jax.numpy as jnp
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init


def main():
    print("=" * 60)
    print("1D Ising Chain with THRML")
    print("=" * 60)
    
    # Model parameters
    n_spins = 10
    coupling_strength = 0.5  # J (positive = ferromagnetic)
    external_field = 0.0     # h (no external field)
    temperature = 1.0        # T (units where k_B = 1)
    beta = 1.0 / temperature  # Inverse temperature
    
    print(f"\nModel Configuration:")
    print(f"  Number of spins: {n_spins}")
    print(f"  Coupling J: {coupling_strength}")
    print(f"  External field h: {external_field}")
    print(f"  Temperature T: {temperature}")
    print(f"  Beta (1/T): {beta}")
    
    # Create the graph structure
    print("\n" + "-" * 60)
    print("Building Graph Structure")
    print("-" * 60)
    
    # Create nodes (one per spin)
    nodes = [SpinNode() for _ in range(n_spins)]
    print(f"Created {len(nodes)} spin nodes")
    
    # Create edges (connect adjacent spins in a chain)
    edges = [(nodes[i], nodes[i+1]) for i in range(n_spins-1)]
    print(f"Created {len(edges)} edges (linear chain)")
    
    # Define energy function parameters
    biases = jnp.zeros((n_spins,))  # No site-specific biases
    weights = jnp.ones((n_spins-1,)) * coupling_strength  # Uniform coupling
    
    # Create the Ising model
    model = IsingEBM(nodes, edges, biases, weights, jnp.array(beta))
    print(f"Created IsingEBM model")
    
    # Define block structure for Gibbs sampling
    print("\n" + "-" * 60)
    print("Block Gibbs Sampling Structure")
    print("-" * 60)
    
    # For 1D chain, we can use 2-coloring:
    # Block 0: nodes at even indices (0, 2, 4, ...)
    # Block 1: nodes at odd indices (1, 3, 5, ...)
    # These blocks don't interact within themselves, so can be updated in parallel
    
    free_blocks = [
        Block(nodes[::2]),   # Even-indexed nodes
        Block(nodes[1::2])   # Odd-indexed nodes
    ]
    
    print(f"Block 0 (even): {len(free_blocks[0].nodes)} nodes")
    print(f"Block 1 (odd): {len(free_blocks[1].nodes)} nodes")
    print("These blocks can be updated in parallel!")
    
    # Create the sampling program
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
    print(f"Created sampling program")
    
    # Run sampling
    print("\n" + "-" * 60)
    print("Running Gibbs Sampling")
    print("-" * 60)
    
    key = jax.random.key(42)
    k_init, k_samp = jax.random.split(key, 2)
    
    # Initialize state randomly
    init_state = hinton_init(k_init, model, free_blocks, ())
    print("Initialized random starting state")
    
    # Define sampling schedule
    n_warmup = 100      # Burn-in steps to reach equilibrium
    n_samples = 1000    # Number of samples to collect
    steps_per_sample = 2  # Gibbs steps between samples
    
    schedule = SamplingSchedule(
        n_warmup=n_warmup,
        n_samples=n_samples,
        steps_per_sample=steps_per_sample
    )
    
    print(f"Schedule: {n_warmup} warmup + {n_samples} samples")
    print(f"Running sampling... ", end="", flush=True)
    
    # Sample!
    samples = sample_states(
        k_samp, 
        program, 
        schedule, 
        init_state, 
        [], 
        [Block(nodes)]
    )
    
    print("Done!")
    
    # Analyze results
    print("\n" + "-" * 60)
    print("Results")
    print("-" * 60)
    
    # samples[0] has shape (n_samples, n_spins)
    spin_samples = samples[0]
    print(f"Sample array shape: {spin_samples.shape}")
    
    # Compute statistics
    mean_magnetization = jnp.mean(spin_samples)
    magnetization_per_sample = jnp.mean(spin_samples, axis=1)
    abs_magnetization = jnp.mean(jnp.abs(magnetization_per_sample))
    
    print(f"\nStatistics:")
    print(f"  Mean magnetization: {mean_magnetization:.4f}")
    print(f"  |Magnetization|: {abs_magnetization:.4f}")
    print(f"  Std deviation: {jnp.std(magnetization_per_sample):.4f}")
    
    # Show first few samples
    print(f"\nFirst 5 samples:")
    for i in range(min(5, n_samples)):
        spins_str = ''.join(['↑' if s > 0 else '↓' for s in spin_samples[i]])
        mag = magnetization_per_sample[i]
        print(f"  Sample {i+1}: {spins_str}  (m={mag:.2f})")
    
    print("\n" + "=" * 60)
    print("Success! ✓ (Now running on GPU!)")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Try changing temperature and see how magnetization changes")
    print("  - Increase coupling strength (J) to see stronger alignment")
    print("  - Add external field (h) to bias one direction")
    print("  - Run ising_2d.py for 2D lattice visualization")


if __name__ == "__main__":
    main()
