#!/usr/bin/env python3
"""
MNIST Data Exploration
======================

Phase 1, Step 1.1: Download & Explore MNIST

Goals:
- Download MNIST dataset (60k train, 10k test)
- Visualize sample digits
- Analyze pixel value distributions
- Decide on discretization strategy (binary, 4-level, or 16-level)
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def download_mnist(data_dir='./data'):
    """Download MNIST dataset."""
    print("=" * 70)
    print("DOWNLOADING MNIST DATASET")
    print("=" * 70)
    
    # Create data directory
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Download training set
    print("\n📥 Downloading training set...")
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    
    # Download test set
    print("📥 Downloading test set...")
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    
    print(f"\n✓ Training samples: {len(train_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    
    return train_dataset, test_dataset


def visualize_samples(dataset, n_samples=20, save_path='mnist_samples.png'):
    """Visualize random samples from dataset."""
    print("\n" + "=" * 70)
    print("VISUALIZING SAMPLE DIGITS")
    print("=" * 70)
    
    # Randomly select samples
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    
    # Create subplot grid
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        image = image.squeeze().numpy()
        
        axes[i].imshow(image, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f'Label: {label}', fontsize=12, weight='bold')
        axes[i].axis('off')
    
    plt.suptitle('MNIST Sample Digits', fontsize=16, weight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {save_path}")
    plt.close()


def analyze_pixel_distributions(dataset, n_samples=10000):
    """Analyze pixel value distributions."""
    print("\n" + "=" * 70)
    print("ANALYZING PIXEL VALUE DISTRIBUTIONS")
    print("=" * 70)
    
    # Sample random images
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    
    # Collect pixel values
    all_pixels = []
    for idx in indices:
        image, _ = dataset[idx]
        all_pixels.append(image.numpy().flatten())
    
    all_pixels = np.concatenate(all_pixels)
    
    # Statistics
    print(f"\n📊 Statistics from {n_samples} images:")
    print(f"  • Total pixels analyzed: {len(all_pixels):,}")
    print(f"  • Min value: {all_pixels.min():.6f}")
    print(f"  • Max value: {all_pixels.max():.6f}")
    print(f"  • Mean value: {all_pixels.mean():.6f}")
    print(f"  • Std deviation: {all_pixels.std():.6f}")
    print(f"  • Median value: {np.median(all_pixels):.6f}")
    
    # Histogram analysis
    print("\n📈 Distribution Analysis:")
    
    # Count near-zero and near-one values
    near_zero = np.sum(all_pixels < 0.1)
    near_one = np.sum(all_pixels > 0.9)
    middle = len(all_pixels) - near_zero - near_one
    
    print(f"  • Near 0 (< 0.1): {near_zero:,} ({100*near_zero/len(all_pixels):.1f}%)")
    print(f"  • Middle (0.1-0.9): {middle:,} ({100*middle/len(all_pixels):.1f}%)")
    print(f"  • Near 1 (> 0.9): {near_one:,} ({100*near_one/len(all_pixels):.1f}%)")
    
    # Plot histogram
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Full distribution
    axes[0].hist(all_pixels, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Pixel Value', fontsize=12, weight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, weight='bold')
    axes[0].set_title('Full Distribution', fontsize=14, weight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Log scale
    axes[1].hist(all_pixels, bins=100, color='coral', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Pixel Value', fontsize=12, weight='bold')
    axes[1].set_ylabel('Frequency (log scale)', fontsize=12, weight='bold')
    axes[1].set_title('Log Scale Distribution', fontsize=14, weight='bold')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    # Non-zero values only
    non_zero = all_pixels[all_pixels > 0.01]
    axes[2].hist(non_zero, bins=50, color='seagreen', edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Pixel Value', fontsize=12, weight='bold')
    axes[2].set_ylabel('Frequency', fontsize=12, weight='bold')
    axes[2].set_title('Non-Zero Values Only', fontsize=14, weight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pixel_distributions.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved distribution plots to: pixel_distributions.png")
    plt.close()
    
    return all_pixels


def analyze_discretization_strategies(pixels):
    """Compare different discretization strategies."""
    print("\n" + "=" * 70)
    print("DISCRETIZATION STRATEGY ANALYSIS")
    print("=" * 70)
    
    # Binary discretization (2 levels)
    binary = (pixels > 0.5).astype(int)
    binary_unique = np.unique(binary, return_counts=True)
    
    print("\n🔲 BINARY (2 levels: 0, 1)")
    print(f"  Threshold: 0.5")
    print(f"  Level 0: {binary_unique[1][0]:,} pixels ({100*binary_unique[1][0]/len(pixels):.1f}%)")
    print(f"  Level 1: {binary_unique[1][1]:,} pixels ({100*binary_unique[1][1]/len(pixels):.1f}%)")
    print(f"  Information loss: HIGH (but simplest for Ising-like models)")
    
    # 4-level discretization
    levels_4 = np.digitize(pixels, bins=[0.25, 0.5, 0.75]) 
    levels_4_unique = np.unique(levels_4, return_counts=True)
    
    print("\n🔲 4-LEVEL (0, 1, 2, 3)")
    print(f"  Bins: [0-0.25), [0.25-0.5), [0.5-0.75), [0.75-1.0]")
    for level, count in zip(levels_4_unique[0], levels_4_unique[1]):
        print(f"  Level {level}: {count:,} pixels ({100*count/len(pixels):.1f}%)")
    print(f"  Information loss: MODERATE (good balance)")
    
    # 16-level discretization
    levels_16 = (pixels * 15).astype(int)
    levels_16 = np.clip(levels_16, 0, 15)
    
    print("\n🔲 16-LEVEL (0-15)")
    print(f"  Bins: 16 equally-spaced levels")
    print(f"  Unique levels used: {len(np.unique(levels_16))}")
    print(f"  Information loss: LOW (but more complex sampling)")
    
    # Visualize discretization effects
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Get a sample image
    sample_idx = 42
    sample_image, sample_label = None, None
    for i, (img, lbl) in enumerate(torchvision.datasets.MNIST(
        root='./data', train=True, download=False, transform=transforms.ToTensor()
    )):
        if i == sample_idx:
            sample_image = img.squeeze().numpy()
            sample_label = lbl
            break
    
    # Original
    axes[0, 0].imshow(sample_image, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title(f'Original (Label: {sample_label})', fontsize=12, weight='bold')
    axes[0, 0].axis('off')
    
    # Binary
    binary_img = (sample_image > 0.5).astype(float)
    axes[0, 1].imshow(binary_img, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Binary (2 levels)', fontsize=12, weight='bold')
    axes[0, 1].axis('off')
    
    # 4-level
    levels_4_img = np.digitize(sample_image, bins=[0.25, 0.5, 0.75]) / 3.0
    axes[0, 2].imshow(levels_4_img, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('4-Level', fontsize=12, weight='bold')
    axes[0, 2].axis('off')
    
    # 16-level
    levels_16_img = (sample_image * 15).astype(int) / 15.0
    axes[0, 3].imshow(levels_16_img, cmap='gray', vmin=0, vmax=1)
    axes[0, 3].set_title('16-Level', fontsize=12, weight='bold')
    axes[0, 3].axis('off')
    
    # Difference maps
    axes[1, 0].imshow(sample_image, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Reference', fontsize=12, weight='bold')
    axes[1, 0].axis('off')
    
    diff_binary = np.abs(sample_image - binary_img)
    axes[1, 1].imshow(diff_binary, cmap='hot', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Binary Error\n(MAE: {diff_binary.mean():.3f})', fontsize=11, weight='bold')
    axes[1, 1].axis('off')
    
    diff_4 = np.abs(sample_image - levels_4_img)
    axes[1, 2].imshow(diff_4, cmap='hot', vmin=0, vmax=1)
    axes[1, 2].set_title(f'4-Level Error\n(MAE: {diff_4.mean():.3f})', fontsize=11, weight='bold')
    axes[1, 2].axis('off')
    
    diff_16 = np.abs(sample_image - levels_16_img)
    axes[1, 3].imshow(diff_16, cmap='hot', vmin=0, vmax=1)
    axes[1, 3].set_title(f'16-Level Error\n(MAE: {diff_16.mean():.3f})', fontsize=11, weight='bold')
    axes[1, 3].axis('off')
    
    plt.suptitle('Discretization Strategy Comparison', fontsize=16, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('discretization_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved discretization comparison to: discretization_comparison.png")
    plt.close()


def make_recommendation():
    """Provide recommendation for discretization strategy."""
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    
    print("""
🎯 RECOMMENDED STRATEGY: 4-LEVEL DISCRETIZATION

Rationale:
  ✓ Balance between expressiveness and complexity
  ✓ Captures more detail than binary
  ✓ Still simple enough for efficient Gibbs sampling
  ✓ Compatible with CategoricalNode in THRML
  ✓ Small state space (4^784 vs 16^784)

Implementation:
  • Map [0, 1] → {0, 1, 2, 3}
  • Bins: [0-0.25), [0.25-0.5), [0.5-0.75), [0.75-1.0]
  • Can always simplify to binary if needed
  • Can upgrade to 16-level if quality demands it

Alternative Strategies:
  
  📌 Binary (2-level):
     Pros: Simplest, direct Ising model analogy
     Cons: Significant information loss
     Use if: EBM training struggles or want pure Ising comparison
  
  📌 16-level:
     Pros: High fidelity, minimal information loss
     Cons: Larger state space, slower sampling
     Use if: Quality is critical and we have computational budget

Next Steps:
  1. Implement 4-level preprocessing pipeline
  2. Create discretization/undiscretization functions
  3. Prepare data loaders for training
  4. Keep binary version as fallback option
    """)


def main():
    """Run complete data exploration."""
    print("\n" + "=" * 70)
    print("PHASE 1, STEP 1.1: MNIST DATA EXPLORATION")
    print("=" * 70)
    
    # Download data
    train_dataset, test_dataset = download_mnist('./data')
    
    # Visualize samples
    visualize_samples(train_dataset, n_samples=20, save_path='mnist_samples.png')
    
    # Analyze distributions
    pixels = analyze_pixel_distributions(train_dataset, n_samples=10000)
    
    # Compare discretization strategies
    analyze_discretization_strategies(pixels)
    
    # Make recommendation
    make_recommendation()
    
    print("\n" + "=" * 70)
    print("✅ STEP 1.1 COMPLETE")
    print("=" * 70)
    print("\n📁 Generated Files:")
    print("  • mnist_samples.png - Sample digit visualization")
    print("  • pixel_distributions.png - Pixel value histograms")
    print("  • discretization_comparison.png - Strategy comparison")
    print("  • ./data/ - Downloaded MNIST dataset")
    
    print("\n🎯 Decision Made: Use 4-LEVEL discretization")
    print("\n➡️  Next: Step 1.2 - Implement preprocessing pipeline")


if __name__ == "__main__":
    main()
