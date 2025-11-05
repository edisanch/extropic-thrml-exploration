#!/usr/bin/env python3
"""
Test Trained EBM Model
======================

Load the trained model and test if it learned digit 3.
Since we trained with use_gibbs_in_training=False (random negatives),
we'll test by checking if the model assigns lower energy to real 3s vs random noise.

NO GIBBS SAMPLING NEEDED - just energy evaluation!
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ebm_model import CategoricalEBM
from preprocessing import undiscretize_image
from create_cache import load_preprocessed_mnist


# ============================================================================
# ENERGY EVALUATION (NO SAMPLING NEEDED!)
# ============================================================================

def evaluate_energies(model_path: str = './checkpoints/ebm_digit3_final.pt',
                     n_samples: int = 100,
                     seed: int = 42):
    """
    Test if model learned by comparing energies on:
    - Real digit 3s (should have LOW energy)
    - Random noise (should have HIGH energy)
    
    Since we trained with random negatives (not Gibbs sampling),
    we just check if the model learned to discriminate.
    """
    print("=" * 70)
    print("TESTING TRAINED EBM MODEL")
    print("=" * 70)
    
    # Load trained model
    print(f"\n📦 Loading model from: {model_path}")
    ebm = CategoricalEBM.load(model_path)
    ebm.eval()
    
    print("\n📊 Model Parameters:")
    summary = ebm.get_parameter_summary()
    for key, val in summary.items():
        print(f"  {key}: {val:.6f}")
    
    # Load real digit 3 data
    print("\n📥 Loading real digit 3 samples...")
    train_images, train_labels, _, _, _ = load_preprocessed_mnist()
    digit3_mask = (train_labels == 3)
    digit3_images = train_images[digit3_mask]
    
    # Sample random subset
    np.random.seed(seed)
    indices = np.random.choice(len(digit3_images), n_samples, replace=False)
    real_samples = digit3_images[indices]
    
    # Generate random noise samples
    print(f"🎲 Generating {n_samples} random noise samples...")
    random_samples = np.random.randint(0, 4, size=(n_samples, 28, 28))
    
    # Convert to torch
    real_torch = torch.from_numpy(real_samples).long()
    random_torch = torch.from_numpy(random_samples).long()
    
    # Compute energies
    print("\n⚡ Computing energies (fast - no sampling!)...")
    with torch.no_grad():
        energy_real = ebm(real_torch).numpy()
        energy_random = ebm(random_torch).numpy()
    
    # Statistics
    print("\n" + "=" * 70)
    print("ENERGY STATISTICS")
    print("=" * 70)
    
    print("\n📊 Real Digit 3 (should be LOW energy):")
    print(f"   Mean: {energy_real.mean():.2f}")
    print(f"   Std:  {energy_real.std():.2f}")
    print(f"   Min:  {energy_real.min():.2f}")
    print(f"   Max:  {energy_real.max():.2f}")
    
    print("\n📊 Random Noise (should be HIGH energy):")
    print(f"   Mean: {energy_random.mean():.2f}")
    print(f"   Std:  {energy_random.std():.2f}")
    print(f"   Min:  {energy_random.min():.2f}")
    print(f"   Max:  {energy_random.max():.2f}")
    
    energy_gap = energy_random.mean() - energy_real.mean()
    print(f"\n⚖️  Energy Gap (random - real): {energy_gap:.2f}")
    
    if energy_gap > 0:
        print("   ✅ GOOD! Model assigns lower energy to real data than noise")
        print(f"   Model learned to discriminate (gap = {energy_gap:.2f})")
    else:
        print("   ❌ BAD! Model assigns higher energy to real data")
        print("   Model did not learn properly")
    
    # Visualize energy distributions
    visualize_energy_distributions(energy_real, energy_random)
    
    # Show example images
    visualize_samples(real_samples[:16], "Real Digit 3 Samples")
    visualize_samples(random_samples[:16], "Random Noise Samples")
    
    return energy_real, energy_random


def visualize_energy_distributions(energy_real, energy_random):
    """Plot histogram of energy distributions."""
    plt.figure(figsize=(10, 6))
    
    plt.hist(energy_real, bins=30, alpha=0.7, label='Real Digit 3', color='blue')
    plt.hist(energy_random, bins=30, alpha=0.7, label='Random Noise', color='red')
    
    plt.axvline(energy_real.mean(), color='blue', linestyle='--', 
                label=f'Real Mean: {energy_real.mean():.1f}')
    plt.axvline(energy_random.mean(), color='red', linestyle='--',
                label=f'Random Mean: {energy_random.mean():.1f}')
    
    plt.xlabel('Energy')
    plt.ylabel('Count')
    plt.title('Energy Distribution: Real vs Random')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('energy_distribution.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved energy distribution to: energy_distribution.png")
    plt.close()


def visualize_samples(samples, title):
    """Visualize a grid of samples."""
    n_samples = len(samples)
    grid_size = int(np.ceil(np.sqrt(n_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    fig.suptitle(title, fontsize=16)
    
    for idx in range(grid_size * grid_size):
        row = idx // grid_size
        col = idx % grid_size
        ax = axes[row, col] if grid_size > 1 else axes
        
        if idx < n_samples:
            # Undiscretize for visualization
            img = undiscretize_image(samples[idx], n_levels=4)
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        else:
            ax.axis('off')
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    filename = title.lower().replace(' ', '_') + '.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved {title} to: {filename}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Test the trained model."""
    print("\n" + "=" * 70)
    print("PHASE 3.1: TESTING TRAINED MODEL")
    print("=" * 70)
    
    print("\n📝 Note:")
    print("   During training, we used use_gibbs_in_training=False")
    print("   This means we trained with random negatives (not proper CD-1)")
    print("   So we'll test by checking if model learned to discriminate")
    print("   Real digit 3 should have LOWER energy than random noise")
    print("\n   ⚡ No Gibbs sampling needed - just energy evaluation!")
    
    energy_real, energy_random = evaluate_energies(
        model_path='./checkpoints/ebm_digit3_final.pt',
        n_samples=100,
        seed=42
    )
    
    print("\n" + "=" * 70)
    print("✅ TESTING COMPLETE")
    print("=" * 70)
    
    print("\n📊 Summary:")
    print(f"   Real digit 3 energy: {energy_real.mean():.2f} ± {energy_real.std():.2f}")
    print(f"   Random noise energy: {energy_random.mean():.2f} ± {energy_random.std():.2f}")
    print(f"   Energy gap: {energy_random.mean() - energy_real.mean():.2f}")
    
    print("\n💡 Next Steps:")
    print("   1. If gap is positive → Model learned! But used random negatives")
    print("   2. For proper CD-1, need to optimize Gibbs sampling (Phase 4)")
    print("   3. Then retrain with use_gibbs_in_training=True")
    print("   4. Then can actually GENERATE new digit 3s (not just discriminate)")


if __name__ == "__main__":
    main()
