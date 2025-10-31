# Ising Model Phase Transition Explorer

Visualize and explore phase transitions in 2D Ising models using THRML's block Gibbs sampling.

## Overview

The **Ising model** is a fundamental model in statistical mechanics that describes magnetism. It consists of spins (think tiny magnets) on a lattice that can point up (+1) or down (-1). Spins want to align with their neighbors (ferromagnetic coupling).

At **high temperatures**: Spins are random (no magnetization)  
At **low temperatures**: Spins align spontaneously (magnetization appears)  
At **critical temperature**: Phase transition occurs!

This is the exact type of problem that Extropic's TSUs are designed to accelerate.

## Physics Background

### Energy Function (Hamiltonian)
```
E = -J * Σ(s_i * s_j) - h * Σ(s_i)
```
- `J`: Coupling strength (positive = ferromagnetic)
- `h`: External magnetic field
- `s_i`: Spin at site i (+1 or -1)

### Boltzmann Distribution
```
P(s) ∝ exp(-β * E(s))
```
- `β = 1/T`: Inverse temperature
- Higher T → more random states
- Lower T → prefer low-energy aligned states

### Block Gibbs Sampling
Update spins in parallel blocks using conditional probability:
```
P(s_i = +1 | neighbors) = 1 / (1 + exp(-2β * h_eff))
h_eff = J * Σ(neighbors) + h
```

## What We'll Explore

1. **Basic Sampling**: Implement 2D Ising model with THRML
2. **Phase Transition**: Sweep temperature and measure magnetization
3. **Block Structure**: Compare different graph colorings
4. **Visualization**: Animate spin dynamics in real-time
5. **Benchmarking**: Compare THRML vs naive Python implementation

## Files

- `ising_basic.py` - Simple 1D Ising chain (start here)
- `ising_2d.py` - 2D lattice with visualization
- `phase_transition.py` - Temperature sweep and magnetization curves
- `block_comparison.py` - Different coloring strategies
- `visualize_sampling.py` - Animated spin dynamics

## Running

```bash
# Activate virtual environment
source ../.venv/bin/activate

# Simple 1D example
python ising_basic.py

# 2D visualization
python ising_2d.py

# Phase transition analysis
python phase_transition.py
```

## Learning Goals

- **Understand block Gibbs sampling** at a fundamental level
- **See how TSU hardware would parallelize** the spin updates
- **Visualize statistical mechanics** concepts (phase transitions, spontaneous symmetry breaking)
- **Build intuition** for when sampling is the bottleneck vs compute

## Next Steps

After mastering the Ising model, you'll be ready to:
- Move to more complex EBMs (Potts models, RBMs)
- Train generative models on real data (MNIST)
- Understand Extropic's DTM algorithm
- Evaluate their efficiency claims critically

---

*"The Ising model is to statistical physics what the harmonic oscillator is to quantum mechanics - a simple model that teaches you everything."*
