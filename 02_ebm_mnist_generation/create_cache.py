#!/usr/bin/env python3
"""
Cache Preprocessed MNIST Data
==============================

Create and save discretized MNIST arrays to disk for faster loading.
This avoids re-discretizing images every time we load the data.
"""

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from preprocessing import discretize_image, DiscretizationConfig
import pickle
from tqdm import tqdm


def create_preprocessed_cache(data_dir='./data', cache_dir='./preprocessed_cache'):
    """
    Create cached versions of discretized MNIST data.
    
    Saves:
    - train_images.npy: (60000, 28, 28) discrete training images
    - train_labels.npy: (60000,) training labels
    - test_images.npy: (10000, 28, 28) discrete test images  
    - test_labels.npy: (10000,) test labels
    - config.pkl: discretization configuration
    """
    print("=" * 70)
    print("CREATING PREPROCESSED DATA CACHE")
    print("=" * 70)
    
    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Cache directory: {cache_path.absolute()}")
    print(f"⚙️  Configuration:")
    print(f"  • Discretization levels: {DiscretizationConfig.N_LEVELS}")
    print(f"  • Bin edges: {DiscretizationConfig.BINS}")
    
    # Load original MNIST
    print("\n📥 Loading training data...")
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=False,
        transform=transforms.ToTensor()
    )
    
    print("📥 Loading test data...")
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=False,
        transform=transforms.ToTensor()
    )
    
    # Discretize training data
    print(f"\n🔢 Discretizing {len(train_dataset)} training images...")
    train_images = []
    train_labels = []
    
    for image, label in tqdm(train_dataset, desc="Training", ncols=70):
        image = image.squeeze(0)  # Remove channel dimension
        discrete = discretize_image(image)
        train_images.append(discrete.numpy())
        train_labels.append(label)
    
    train_images = np.stack(train_images)
    train_labels = np.array(train_labels)
    
    # Discretize test data
    print(f"🔢 Discretizing {len(test_dataset)} test images...")
    test_images = []
    test_labels = []
    
    for image, label in tqdm(test_dataset, desc="Testing", ncols=70):
        image = image.squeeze(0)
        discrete = discretize_image(image)
        test_images.append(discrete.numpy())
        test_labels.append(label)
    
    test_images = np.stack(test_images)
    test_labels = np.array(test_labels)
    
    # Save to disk
    print("\n💾 Saving preprocessed data...")
    
    train_images_path = cache_path / 'train_images.npy'
    train_labels_path = cache_path / 'train_labels.npy'
    test_images_path = cache_path / 'test_images.npy'
    test_labels_path = cache_path / 'test_labels.npy'
    config_path = cache_path / 'config.pkl'
    
    np.save(train_images_path, train_images)
    np.save(train_labels_path, train_labels)
    np.save(test_images_path, test_images)
    np.save(test_labels_path, test_labels)
    
    # Save configuration
    config = {
        'n_levels': DiscretizationConfig.N_LEVELS,
        'bins': DiscretizationConfig.BINS,
        'bin_centers': DiscretizationConfig.get_bin_centers(),
        'image_shape': (28, 28),
        'n_train': len(train_images),
        'n_test': len(test_images)
    }
    
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    
    # Report sizes
    train_size = train_images_path.stat().st_size / (1024**2)
    test_size = test_images_path.stat().st_size / (1024**2)
    total_size = train_size + test_size
    
    print(f"\n✓ Files saved:")
    print(f"  • {train_images_path.name}: {train_size:.1f} MB")
    print(f"  • {train_labels_path.name}: {train_labels_path.stat().st_size / 1024:.1f} KB")
    print(f"  • {test_images_path.name}: {test_size:.1f} MB")
    print(f"  • {test_labels_path.name}: {test_labels_path.stat().st_size / 1024:.1f} KB")
    print(f"  • {config_path.name}: {config_path.stat().st_size / 1024:.1f} KB")
    print(f"\n  Total cache size: {total_size:.1f} MB")
    
    # Verify data
    print("\n🔍 Verifying cached data...")
    print(f"  • Train images shape: {train_images.shape}")
    print(f"  • Train labels shape: {train_labels.shape}")
    print(f"  • Test images shape: {test_images.shape}")
    print(f"  • Test labels shape: {test_labels.shape}")
    print(f"  • Train value range: [{train_images.min()}, {train_images.max()}]")
    print(f"  • Test value range: [{test_images.min()}, {test_images.max()}]")
    
    # Show class distribution
    print("\n📊 Label distribution (training set):")
    for digit in range(10):
        count = np.sum(train_labels == digit)
        print(f"  Digit {digit}: {count:5d} samples")
    
    print("\n" + "=" * 70)
    print("✅ PREPROCESSED CACHE CREATED")
    print("=" * 70)
    print(f"\n📁 Cache location: {cache_path.absolute()}")
    print("\nTo use this cache in your code:")
    print("  from load_cached_data import load_preprocessed_mnist")
    print("  train_images, train_labels, test_images, test_labels = load_preprocessed_mnist()")


def load_preprocessed_mnist(cache_dir='./preprocessed_cache'):
    """
    Load preprocessed MNIST data from cache.
    
    Returns:
        train_images: (60000, 28, 28) numpy array
        train_labels: (60000,) numpy array
        test_images: (10000, 28, 28) numpy array
        test_labels: (10000,) numpy array
        config: dict with discretization configuration
    """
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache directory not found: {cache_path.absolute()}\n"
            "Run create_preprocessed_cache() first."
        )
    
    print(f"📂 Loading preprocessed data from: {cache_path.absolute()}")
    
    train_images = np.load(cache_path / 'train_images.npy')
    train_labels = np.load(cache_path / 'train_labels.npy')
    test_images = np.load(cache_path / 'test_images.npy')
    test_labels = np.load(cache_path / 'test_labels.npy')
    
    with open(cache_path / 'config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    print(f"✓ Loaded {len(train_images)} training + {len(test_images)} test images")
    
    return train_images, train_labels, test_images, test_labels, config


def verify_cache_integrity(cache_dir='./preprocessed_cache'):
    """Verify that cached data is valid."""
    print("\n🔍 Verifying cache integrity...")
    
    try:
        train_images, train_labels, test_images, test_labels, config = \
            load_preprocessed_mnist(cache_dir)
        
        # Check shapes
        assert train_images.shape == (60000, 28, 28), "Invalid train images shape"
        assert train_labels.shape == (60000,), "Invalid train labels shape"
        assert test_images.shape == (10000, 28, 28), "Invalid test images shape"
        assert test_labels.shape == (10000,), "Invalid test labels shape"
        
        # Check value ranges
        assert train_images.min() >= 0, "Train images have negative values"
        assert train_images.max() < config['n_levels'], "Train images exceed max level"
        assert test_images.min() >= 0, "Test images have negative values"
        assert test_images.max() < config['n_levels'], "Test images exceed max level"
        
        # Check labels
        assert train_labels.min() >= 0 and train_labels.max() <= 9, "Invalid train labels"
        assert test_labels.min() >= 0 and test_labels.max() <= 9, "Invalid test labels"
        
        print("✓ Cache integrity verified - all checks passed!")
        return True
        
    except Exception as e:
        print(f"❌ Cache verification failed: {e}")
        return False


def main():
    """Create and verify preprocessed cache."""
    # Create cache
    create_preprocessed_cache(
        data_dir='./data',
        cache_dir='./preprocessed_cache'
    )
    
    # Verify it works
    verify_cache_integrity('./preprocessed_cache')
    
    print("\n" + "=" * 70)
    print("STEP 1.2 NOW TRULY COMPLETE ✅")
    print("=" * 70)
    print("\n➡️  Ready for Phase 2: EBM Architecture Design")


if __name__ == "__main__":
    main()
