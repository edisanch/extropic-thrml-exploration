#!/usr/bin/env python3
"""
MNIST Preprocessing Pipeline
=============================

Phase 1, Step 1.2: Preprocessing Pipeline

Goals:
- Implement discretization function (continuous → discrete pixels)
- Implement undiscretization function (discrete → continuous for visualization)
- Create train/validation splits
- Prepare data loaders for training
- Cache preprocessed data for efficiency
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import pickle
from torch.utils.data import Dataset, DataLoader, random_split


class DiscretizationConfig:
    """Configuration for discretization strategy."""
    
    # 4-level discretization (recommended from Step 1.1)
    N_LEVELS = 4
    BINS = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Alternative: Binary discretization (fallback)
    # N_LEVELS = 2
    # BINS = [0.0, 0.5, 1.0]
    
    # Alternative: 16-level discretization (high quality)
    # N_LEVELS = 16
    # BINS = [i/16.0 for i in range(17)]
    
    @classmethod
    def get_bin_centers(cls):
        """Get center values of each bin for undiscretization."""
        centers = []
        for i in range(len(cls.BINS) - 1):
            center = (cls.BINS[i] + cls.BINS[i+1]) / 2.0
            centers.append(center)
        return centers


def discretize_image(image, n_levels=None):
    """
    Convert continuous pixel values [0, 1] to discrete levels.
    
    Args:
        image: torch.Tensor or numpy array with values in [0, 1]
        n_levels: number of discrete levels (uses config default if None)
    
    Returns:
        Discretized image with integer values in {0, 1, ..., n_levels-1}
    """
    if n_levels is None:
        n_levels = DiscretizationConfig.N_LEVELS
    
    bins = DiscretizationConfig.BINS
    
    # Convert to numpy if needed
    is_torch = isinstance(image, torch.Tensor)
    if is_torch:
        img = image.numpy()
    else:
        img = image
    
    # Digitize: assigns each pixel to a bin
    # digitize returns 1-indexed bins, so subtract 1 to get 0-indexed levels
    discrete = np.digitize(img, bins=bins[1:-1])  # Use interior bin edges
    discrete = np.clip(discrete, 0, n_levels - 1)
    
    if is_torch:
        return torch.from_numpy(discrete).long()
    return discrete.astype(np.int64)


def undiscretize_image(discrete_image, n_levels=None):
    """
    Convert discrete levels back to continuous pixel values [0, 1].
    Uses bin center values for reconstruction.
    
    Args:
        discrete_image: torch.Tensor or numpy array with integer values
        n_levels: number of discrete levels (uses config default if None)
    
    Returns:
        Continuous image with values in [0, 1]
    """
    if n_levels is None:
        n_levels = DiscretizationConfig.N_LEVELS
    
    centers = DiscretizationConfig.get_bin_centers()
    
    # Convert to numpy if needed
    is_torch = isinstance(discrete_image, torch.Tensor)
    if is_torch:
        img = discrete_image.numpy()
    else:
        img = discrete_image
    
    # Map each level to its bin center
    continuous = np.zeros_like(img, dtype=np.float32)
    for level in range(n_levels):
        mask = (img == level)
        continuous[mask] = centers[level]
    
    if is_torch:
        return torch.from_numpy(continuous).float()
    return continuous.astype(np.float32)


class DiscreteMNIST(Dataset):
    """MNIST dataset with discretized pixel values."""
    
    def __init__(self, root='./data', train=True, discretize=True, download=True):
        """
        Args:
            root: data directory
            train: use training set (True) or test set (False)
            discretize: apply discretization (True) or keep continuous (False)
            download: download data if not present
        """
        self.discretize_flag = discretize
        
        # Load original MNIST
        self.mnist = torchvision.datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.ToTensor()
        )
    
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: discretized image (H, W) with integer values if discretize=True,
                   or continuous image (1, H, W) if discretize=False
            label: digit label (0-9)
        """
        image, label = self.mnist[idx]
        
        if self.discretize_flag:
            # Remove channel dimension and discretize
            image = image.squeeze(0)  # (1, H, W) -> (H, W)
            image = discretize_image(image)
        
        return image, label


def create_data_loaders(batch_size=64, val_split=0.1, discretize=True, 
                       root='./data', num_workers=0):
    """
    Create train, validation, and test data loaders.
    
    Args:
        batch_size: batch size for training
        val_split: fraction of training data to use for validation
        discretize: whether to discretize images
        root: data directory
        num_workers: number of worker processes for data loading
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print("=" * 70)
    print("CREATING DATA LOADERS")
    print("=" * 70)
    
    # Load full training dataset
    train_dataset = DiscreteMNIST(root=root, train=True, discretize=discretize)
    
    # Split into train and validation
    n_total = len(train_dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    
    train_subset, val_subset = random_split(
        train_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Load test dataset
    test_dataset = DiscreteMNIST(root=root, train=False, discretize=discretize)
    
    print(f"\n📊 Dataset Splits:")
    print(f"  • Training samples: {n_train:,}")
    print(f"  • Validation samples: {n_val:,}")
    print(f"  • Test samples: {len(test_dataset):,}")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Discretized: {discretize}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n✓ Data loaders created")
    print(f"  • Train batches: {len(train_loader)}")
    print(f"  • Val batches: {len(val_loader)}")
    print(f"  • Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def save_preprocessed_cache(train_loader, val_loader, test_loader, 
                            cache_path='preprocessed_cache.pkl'):
    """
    Cache preprocessed data loaders to disk for faster loading.
    
    Args:
        train_loader, val_loader, test_loader: data loaders to cache
        cache_path: path to save cache file
    """
    print(f"\n💾 Caching preprocessed data to: {cache_path}")
    
    cache = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'config': {
            'n_levels': DiscretizationConfig.N_LEVELS,
            'bins': DiscretizationConfig.BINS
        }
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    
    print(f"✓ Cache saved ({Path(cache_path).stat().st_size / 1024:.1f} KB)")


def load_preprocessed_cache(cache_path='preprocessed_cache.pkl'):
    """
    Load cached preprocessed data loaders from disk.
    
    Args:
        cache_path: path to cache file
    
    Returns:
        train_loader, val_loader, test_loader or None if cache doesn't exist
    """
    if not Path(cache_path).exists():
        return None
    
    print(f"📂 Loading preprocessed data from cache: {cache_path}")
    
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    
    print(f"✓ Cache loaded")
    return cache['train_loader'], cache['val_loader'], cache['test_loader']


def test_discretization():
    """Test discretization and undiscretization functions."""
    print("\n" + "=" * 70)
    print("TESTING DISCRETIZATION FUNCTIONS")
    print("=" * 70)
    
    # Load a sample image
    dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=False, transform=transforms.ToTensor()
    )
    
    sample_image, sample_label = dataset[0]
    sample_image = sample_image.squeeze(0)  # Remove channel dimension
    
    print(f"\n🖼️  Original Image:")
    print(f"  • Shape: {sample_image.shape}")
    print(f"  • Data type: {sample_image.dtype}")
    print(f"  • Value range: [{sample_image.min():.4f}, {sample_image.max():.4f}]")
    print(f"  • Label: {sample_label}")
    
    # Discretize
    discrete = discretize_image(sample_image)
    print(f"\n🔢 Discretized Image:")
    print(f"  • Shape: {discrete.shape}")
    print(f"  • Data type: {discrete.dtype}")
    print(f"  • Value range: [{discrete.min()}, {discrete.max()}]")
    print(f"  • Unique values: {torch.unique(discrete).tolist()}")
    
    # Count pixels per level
    for level in range(DiscretizationConfig.N_LEVELS):
        count = (discrete == level).sum().item()
        pct = 100 * count / discrete.numel()
        print(f"    Level {level}: {count:5d} pixels ({pct:5.1f}%)")
    
    # Undiscretize
    reconstructed = undiscretize_image(discrete)
    print(f"\n🔄 Reconstructed Image:")
    print(f"  • Shape: {reconstructed.shape}")
    print(f"  • Data type: {reconstructed.dtype}")
    print(f"  • Value range: [{reconstructed.min():.4f}, {reconstructed.max():.4f}]")
    
    # Compute reconstruction error
    error = torch.abs(sample_image - reconstructed)
    print(f"\n📊 Reconstruction Quality:")
    print(f"  • Mean Absolute Error: {error.mean():.6f}")
    print(f"  • Max Absolute Error: {error.max():.6f}")
    print(f"  • PSNR: {-10 * torch.log10(torch.mean(error**2)):.2f} dB")
    
    # Visualize
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(sample_image.numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'Original (Label: {sample_label})', fontsize=14, weight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(discrete.numpy(), cmap='gray', vmin=0, vmax=DiscretizationConfig.N_LEVELS-1)
    axes[1].set_title(f'Discretized ({DiscretizationConfig.N_LEVELS} levels)', 
                     fontsize=14, weight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(reconstructed.numpy(), cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Reconstructed', fontsize=14, weight='bold')
    axes[2].axis('off')
    
    axes[3].imshow(error.numpy(), cmap='hot', vmin=0, vmax=0.5)
    axes[3].set_title(f'Error (MAE: {error.mean():.4f})', fontsize=14, weight='bold')
    axes[3].axis('off')
    
    plt.suptitle('Discretization → Undiscretization Test', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('discretization_test.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved test visualization to: discretization_test.png")
    plt.close()


def visualize_batch(loader, save_path='batch_sample.png', n_samples=16):
    """Visualize a batch of discretized images."""
    print(f"\n📸 Visualizing batch from data loader...")
    
    # Get one batch
    images, labels = next(iter(loader))
    
    # Select subset
    n_show = min(n_samples, len(images))
    images = images[:n_show]
    labels = labels[:n_show]
    
    # Undiscretize for visualization
    continuous_images = []
    for img in images:
        continuous_images.append(undiscretize_image(img))
    
    # Plot
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(n_show):
        axes[i].imshow(continuous_images[i], cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f'Label: {labels[i].item()}', fontsize=12, weight='bold')
        axes[i].axis('off')
    
    plt.suptitle(f'Discretized MNIST Batch ({DiscretizationConfig.N_LEVELS} levels)', 
                fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved batch visualization to: {save_path}")
    plt.close()


def main():
    """Run complete preprocessing pipeline."""
    print("\n" + "=" * 70)
    print("PHASE 1, STEP 1.2: PREPROCESSING PIPELINE")
    print("=" * 70)
    
    # Display configuration
    print(f"\n⚙️  Configuration:")
    print(f"  • Discretization levels: {DiscretizationConfig.N_LEVELS}")
    print(f"  • Bin edges: {DiscretizationConfig.BINS}")
    print(f"  • Bin centers: {DiscretizationConfig.get_bin_centers()}")
    
    # Test discretization functions
    test_discretization()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=64,
        val_split=0.1,
        discretize=True,
        num_workers=0  # Set to 2-4 if you have multicore CPU
    )
    
    # Visualize a batch
    visualize_batch(train_loader, save_path='batch_sample.png', n_samples=16)
    
    # Note: We skip caching data loaders because they contain generator state
    # and are quick to recreate. We'll load fresh each time.
    
    print("\n" + "=" * 70)
    print("✅ STEP 1.2 COMPLETE")
    print("=" * 70)
    
    print("\n📁 Generated Files:")
    print("  • discretization_test.png - Function validation")
    print("  • batch_sample.png - Sample batch visualization")
    print("  • preprocessing.py - Complete preprocessing pipeline")
    
    print("\n🔧 Ready to Use:")
    print("  • discretize_image(img) - Convert continuous → discrete")
    print("  • undiscretize_image(img) - Convert discrete → continuous")
    print("  • DiscreteMNIST dataset class")
    print("  • create_data_loaders() function")
    
    print(f"\n📊 Summary:")
    print(f"  • Training samples: {len(train_loader.dataset):,}")
    print(f"  • Validation samples: {len(val_loader.dataset):,}")
    print(f"  • Test samples: {len(test_loader.dataset):,}")
    print(f"  • Discretization: {DiscretizationConfig.N_LEVELS} levels")
    print(f"  • Image shape: (28, 28)")
    print(f"  • Total variables per image: 784")
    
    print("\n➡️  Next: Phase 2.1 - Define EBM energy function")


if __name__ == "__main__":
    main()
