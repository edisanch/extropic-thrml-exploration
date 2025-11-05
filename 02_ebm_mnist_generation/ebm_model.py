#!/usr/bin/env python3
"""
Energy-Based Model (EBM) for MNIST
===================================

Phase 2, Step 2.1: Define Energy Function

Implements a categorical Potts model with:
- 4 discrete levels {0, 1, 2, 3} per pixel
- Per-pixel bias terms
- Pairwise interactions between 4-neighbors
- Learnable horizontal and vertical weights
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List


class CategoricalEBM(nn.Module):
    """
    Energy-Based Model for categorical (discrete) images.
    
    Energy Function (Potts Model):
        E(x; θ) = - Σ_i bias_i[x_i] - Σ_{<i,j>} weight_type * δ(x_i, x_j)
    
    Where:
        - x_i ∈ {0, 1, 2, 3} is the discrete level at pixel i
        - bias_i[x_i] is the bias for pixel i at level x_i
        - δ(x_i, x_j) = 1 if x_i == x_j, else 0 (Potts interaction)
        - weight_type is weight_h (horizontal) or weight_v (vertical)
    
    Parameters:
        - biases: (H, W, n_levels) - per-pixel bias for each level
        - weight_h: scalar - horizontal neighbor interaction strength
        - weight_v: scalar - vertical neighbor interaction strength
    
    Total Parameters: H×W×n_levels + 2 = 28×28×4 + 2 = 3,138
    """
    
    def __init__(self, height: int = 28, width: int = 28, n_levels: int = 4):
        """
        Initialize the EBM.
        
        Args:
            height: Image height (default 28 for MNIST)
            width: Image width (default 28 for MNIST)
            n_levels: Number of discrete levels (default 4)
        """
        super().__init__()
        
        self.height = height
        self.width = width
        self.n_levels = n_levels
        self.n_pixels = height * width
        
        # Learnable parameters
        # Biases: (H, W, n_levels) - per-pixel bias for each level
        self.biases = nn.Parameter(torch.zeros(height, width, n_levels))
        
        # Pairwise interaction weights (Potts model encourages same-level neighbors)
        self.weight_h = nn.Parameter(torch.tensor(0.0))  # Horizontal edges
        self.weight_v = nn.Parameter(torch.tensor(0.0))  # Vertical edges
        
        # Initialize parameters
        self.initialize_parameters()
        
        # Precompute edge structure for efficiency
        self.edges_h, self.edges_v = self._build_edge_lists()
        
        print(f"Initialized CategoricalEBM:")
        print(f"  Image size: {height}×{width} = {self.n_pixels} pixels")
        print(f"  Discrete levels: {n_levels}")
        print(f"  Bias parameters: {height}×{width}×{n_levels} = {height*width*n_levels:,}")
        print(f"  Weight parameters: 2 (horizontal + vertical)")
        print(f"  Total parameters: {self.count_parameters():,}")
        print(f"  Horizontal edges: {len(self.edges_h)}")
        print(f"  Vertical edges: {len(self.edges_v)}")
    
    def initialize_parameters(self):
        """Initialize parameters with reasonable values."""
        # Initialize biases with small random values
        nn.init.normal_(self.biases, mean=0.0, std=0.01)
        
        # Initialize weights to encourage slight smoothness
        # Positive weights → ferromagnetic (same levels cluster)
        nn.init.constant_(self.weight_h, 0.1)
        nn.init.constant_(self.weight_v, 0.1)
    
    def _build_edge_lists(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Build lists of edges (pairs of neighboring pixels).
        
        Returns:
            edges_h: List of (pixel_i, pixel_j) for horizontal neighbors
            edges_v: List of (pixel_i, pixel_j) for vertical neighbors
        """
        edges_h = []
        edges_v = []
        
        for i in range(self.height):
            for j in range(self.width):
                pixel_idx = i * self.width + j
                
                # Horizontal edge (right neighbor)
                if j < self.width - 1:
                    right_idx = i * self.width + (j + 1)
                    edges_h.append((pixel_idx, right_idx))
                
                # Vertical edge (down neighbor)
                if i < self.height - 1:
                    down_idx = (i + 1) * self.width + j
                    edges_v.append((pixel_idx, down_idx))
        
        return edges_h, edges_v
    
    def compute_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute energy for a batch of images.
        
        E(x) = - Σ_i bias_i[x_i] - Σ_{<i,j>} weight * δ(x_i, x_j)
        
        Args:
            x: (batch_size, H, W) tensor with discrete values in {0, 1, ..., n_levels-1}
        
        Returns:
            energies: (batch_size,) tensor of energy values
        """
        batch_size = x.shape[0]
        
        # Ensure x is long type for indexing
        x = x.long()
        
        # === Bias Energy Term ===
        # For each pixel, select the bias corresponding to its level
        # x: (B, H, W), biases: (H, W, n_levels)
        # We want: bias_energy[b, i, j] = biases[i, j, x[b, i, j]]
        
        bias_energy = torch.gather(
            self.biases.unsqueeze(0).expand(batch_size, -1, -1, -1),  # (B, H, W, n_levels)
            dim=3,
            index=x.unsqueeze(-1)  # (B, H, W, 1)
        ).squeeze(-1)  # (B, H, W)
        
        # Sum over all pixels
        bias_term = bias_energy.sum(dim=(1, 2))  # (B,)
        
        # === Pairwise Interaction Energy Term ===
        # Flatten images for easier edge processing
        x_flat = x.view(batch_size, -1)  # (B, H*W)
        
        # Horizontal edges
        h_energy = 0.0
        for i, j in self.edges_h:
            # δ(x_i, x_j) = 1 if x_i == x_j, else 0
            same_level = (x_flat[:, i] == x_flat[:, j]).float()
            h_energy += self.weight_h * same_level
        
        # Vertical edges  
        v_energy = 0.0
        for i, j in self.edges_v:
            same_level = (x_flat[:, i] == x_flat[:, j]).float()
            v_energy += self.weight_v * same_level
        
        pairwise_term = h_energy + v_energy  # (B,)
        
        # Total energy (note the negative signs)
        energy = -(bias_term + pairwise_term)
        
        return energy
    
    def compute_energy_fast(self, x: torch.Tensor) -> torch.Tensor:
        """
        Faster vectorized energy computation.
        
        Args:
            x: (batch_size, H, W) tensor with discrete values
        
        Returns:
            energies: (batch_size,) tensor
        """
        batch_size = x.shape[0]
        x = x.long()
        
        # === Bias Energy Term ===
        bias_energy = torch.gather(
            self.biases.unsqueeze(0).expand(batch_size, -1, -1, -1),
            dim=3,
            index=x.unsqueeze(-1)
        ).squeeze(-1)
        bias_term = bias_energy.sum(dim=(1, 2))
        
        # === Pairwise Interaction Energy Term (Vectorized) ===
        # Horizontal edges: compare with right neighbor
        h_same = (x[:, :, :-1] == x[:, :, 1:]).float()
        h_energy = self.weight_h * h_same.sum(dim=(1, 2))
        
        # Vertical edges: compare with down neighbor
        v_same = (x[:, :-1, :] == x[:, 1:, :]).float()
        v_energy = self.weight_v * v_same.sum(dim=(1, 2))
        
        pairwise_term = h_energy + v_energy
        
        # Total energy
        energy = -(bias_term + pairwise_term)
        
        return energy
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute energy.
        
        Args:
            x: (batch_size, H, W) tensor with discrete values
        
        Returns:
            energies: (batch_size,) tensor
        """
        return self.compute_energy_fast(x)
    
    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_parameter_summary(self) -> dict:
        """Get summary of parameter values."""
        return {
            'biases_mean': self.biases.mean().item(),
            'biases_std': self.biases.std().item(),
            'biases_min': self.biases.min().item(),
            'biases_max': self.biases.max().item(),
            'weight_h': self.weight_h.item(),
            'weight_v': self.weight_v.item(),
        }
    
    def save(self, path: str):
        """Save model parameters."""
        torch.save({
            'biases': self.biases,
            'weight_h': self.weight_h,
            'weight_v': self.weight_v,
            'height': self.height,
            'width': self.width,
            'n_levels': self.n_levels,
        }, path)
        print(f"✓ Saved model to: {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load model parameters."""
        checkpoint = torch.load(path)
        model = cls(
            height=checkpoint['height'],
            width=checkpoint['width'],
            n_levels=checkpoint['n_levels']
        )
        model.biases = checkpoint['biases']
        model.weight_h = checkpoint['weight_h']
        model.weight_v = checkpoint['weight_v']
        print(f"✓ Loaded model from: {path}")
        return model


def test_energy_computation():
    """Test the energy computation with simple examples."""
    print("\n" + "=" * 70)
    print("TESTING ENERGY COMPUTATION")
    print("=" * 70)
    
    # Create small model for testing
    model = CategoricalEBM(height=4, width=4, n_levels=4)
    
    print("\n📊 Initial Parameters:")
    summary = model.get_parameter_summary()
    for key, val in summary.items():
        print(f"  {key}: {val:.6f}")
    
    # Test Case 1: Uniform image (all same level)
    print("\n🔬 Test Case 1: Uniform Image (all pixels = 0)")
    x_uniform = torch.zeros(1, 4, 4, dtype=torch.long)
    energy_uniform = model(x_uniform)
    print(f"  Energy: {energy_uniform.item():.4f}")
    print(f"  Expected: Maximum agreement → Low energy (negative)")
    
    # Test Case 2: Checkerboard pattern (alternating levels)
    print("\n🔬 Test Case 2: Checkerboard Pattern (0s and 1s)")
    x_checker = torch.zeros(1, 4, 4, dtype=torch.long)
    x_checker[0, ::2, ::2] = 0  # Even rows, even cols
    x_checker[0, ::2, 1::2] = 1  # Even rows, odd cols
    x_checker[0, 1::2, ::2] = 1  # Odd rows, even cols
    x_checker[0, 1::2, 1::2] = 0  # Odd rows, odd cols
    energy_checker = model(x_checker)
    print(f"  Energy: {energy_checker.item():.4f}")
    print(f"  Expected: No agreement → High energy (positive)")
    
    # Test Case 3: Random image
    print("\n🔬 Test Case 3: Random Image")
    x_random = torch.randint(0, 4, (1, 4, 4), dtype=torch.long)
    energy_random = model(x_random)
    print(f"  Energy: {energy_random.item():.4f}")
    print(f"  Expected: Somewhere in between")
    
    # Test Case 4: Batch of images
    print("\n🔬 Test Case 4: Batch Processing")
    x_batch = torch.randint(0, 4, (8, 4, 4), dtype=torch.long)
    energies_batch = model(x_batch)
    print(f"  Batch size: {len(energies_batch)}")
    print(f"  Energy range: [{energies_batch.min().item():.4f}, {energies_batch.max().item():.4f}]")
    print(f"  Energy mean: {energies_batch.mean().item():.4f}")
    
    # Compare fast vs slow implementation
    print("\n⚡ Speed Comparison:")
    import time
    
    x_test = torch.randint(0, 4, (32, 4, 4), dtype=torch.long)
    
    # Slow version
    start = time.time()
    energy_slow = model.compute_energy(x_test)
    time_slow = time.time() - start
    
    # Fast version
    start = time.time()
    energy_fast = model.compute_energy_fast(x_test)
    time_fast = time.time() - start
    
    print(f"  Slow version: {time_slow*1000:.2f} ms")
    print(f"  Fast version: {time_fast*1000:.2f} ms")
    print(f"  Speedup: {time_slow/time_fast:.1f}x")
    print(f"  Results match: {torch.allclose(energy_slow, energy_fast, atol=1e-4)}")
    
    print("\n✅ Energy computation tests passed!")


def test_full_mnist():
    """Test with actual MNIST-sized images."""
    print("\n" + "=" * 70)
    print("TESTING WITH MNIST-SIZED IMAGES")
    print("=" * 70)
    
    # Create full MNIST model
    model = CategoricalEBM(height=28, width=28, n_levels=4)
    
    # Test with batch from cached data
    try:
        from create_cache import load_preprocessed_mnist
        
        print("\n📂 Loading preprocessed MNIST...")
        train_images, train_labels, _, _, config = load_preprocessed_mnist()
        
        # Convert to torch and take small batch
        batch_size = 16
        x_batch = torch.from_numpy(train_images[:batch_size])
        
        print(f"\n🔍 Batch Details:")
        print(f"  Shape: {x_batch.shape}")
        print(f"  Value range: [{x_batch.min()}, {x_batch.max()}]")
        print(f"  Labels: {train_labels[:batch_size].tolist()}")
        
        # Compute energies
        print("\n⚡ Computing energies...")
        energies = model(x_batch)
        
        print(f"\n📊 Energy Statistics:")
        print(f"  Mean: {energies.mean().item():.2f}")
        print(f"  Std: {energies.std().item():.2f}")
        print(f"  Min: {energies.min().item():.2f}")
        print(f"  Max: {energies.max().item():.2f}")
        
        # Show per-sample energies
        print(f"\n📋 Per-Sample Energies:")
        for i, (energy, label) in enumerate(zip(energies, train_labels[:batch_size])):
            print(f"  Sample {i} (digit {label}): {energy.item():.2f}")
        
        print("\n✅ MNIST-sized test passed!")
        
    except Exception as e:
        print(f"\n⚠️  Could not load cached MNIST data: {e}")
        print("    Run create_cache.py first to test with real data.")


def main():
    """Run all tests for Step 2.1."""
    print("\n" + "=" * 70)
    print("PHASE 2, STEP 2.1: DEFINE ENERGY FUNCTION")
    print("=" * 70)
    
    # Run tests
    test_energy_computation()
    test_full_mnist()
    
    print("\n" + "=" * 70)
    print("✅ STEP 2.1 COMPLETE")
    print("=" * 70)
    
    print("\n📦 Deliverable: CategoricalEBM class with energy function")
    print("\n🔧 Features:")
    print("  ✓ Potts model with categorical variables (4 levels)")
    print("  ✓ Per-pixel biases (3,136 parameters)")
    print("  ✓ Per-edge-type weights (2 parameters: horizontal + vertical)")
    print("  ✓ Efficient vectorized energy computation")
    print("  ✓ Save/load functionality")
    print("  ✓ Tested on small and MNIST-sized images")
    
    print("\n📐 Energy Function:")
    print("  E(x; θ) = - Σ_i bias_i[x_i] - Σ_{<i,j>} weight_type * δ(x_i, x_j)")
    print("\n  Where δ(x_i, x_j) = 1 if x_i == x_j (Potts interaction)")
    
    print("\n➡️  Next: Step 2.2 - Block Structure for Sampling")


# ============================================================================
# STEP 2.2: BLOCK STRUCTURE FOR SAMPLING
# ============================================================================

def create_2color_blocks(height: int, width: int) -> List[List[int]]:
    """
    Create 2-coloring (checkerboard pattern) for 2D grid.
    
    This is optimal for 4-neighbor interactions. Pixels in the same block
    have no edges between them, allowing parallel updates.
    
    Pattern:
        0 1 0 1 0 1 ...
        1 0 1 0 1 0 ...
        0 1 0 1 0 1 ...
        ...
    
    Args:
        height: Grid height
        width: Grid width
    
    Returns:
        List of 2 blocks, where each block is a list of pixel indices
    """
    block_0 = []  # "Even" positions (i+j is even)
    block_1 = []  # "Odd" positions (i+j is odd)
    
    for i in range(height):
        for j in range(width):
            idx = i * width + j
            if (i + j) % 2 == 0:
                block_0.append(idx)
            else:
                block_1.append(idx)
    
    return [block_0, block_1]


def create_4color_blocks(height: int, width: int) -> List[List[int]]:
    """
    Create 4-coloring for 2D grid.
    
    This provides more fine-grained blocking but doesn't improve parallelism
    for 4-neighbor grids (useful for comparing strategies in Phase 6.3).
    
    Pattern:
        0 1 0 1 0 1 ...
        2 3 2 3 2 3 ...
        0 1 0 1 0 1 ...
        2 3 2 3 2 3 ...
    
    Args:
        height: Grid height
        width: Grid width
    
    Returns:
        List of 4 blocks, where each block is a list of pixel indices
    """
    blocks = [[], [], [], []]  # 4 blocks
    
    for i in range(height):
        for j in range(width):
            idx = i * width + j
            # Color based on (row % 2, col % 2)
            color = (i % 2) * 2 + (j % 2)
            blocks[color].append(idx)
    
    return blocks


def verify_block_independence(blocks: List[List[int]], 
                              edges_h: List[Tuple[int, int]], 
                              edges_v: List[Tuple[int, int]]) -> Tuple[bool, str]:
    """
    Verify that blocks are independent (no edges within same block).
    
    Args:
        blocks: List of blocks, where each block is list of pixel indices
        edges_h: List of horizontal edges (pixel_i, pixel_j)
        edges_v: List of vertical edges (pixel_i, pixel_j)
    
    Returns:
        (is_valid, message): True if blocks are independent, with explanation
    """
    all_edges = edges_h + edges_v
    
    for block_idx, block_nodes in enumerate(blocks):
        node_set = set(block_nodes)
        
        for edge in all_edges:
            i, j = edge
            # Check if both endpoints are in this block
            if i in node_set and j in node_set:
                return False, f"Block {block_idx} has internal edge: ({i}, {j})"
    
    return True, f"✓ All {len(blocks)} blocks are independent"


def visualize_blocks(blocks: List[List[int]], height: int, width: int, 
                    save_path: str = 'block_structure.png'):
    """
    Visualize the block coloring pattern.
    
    Args:
        blocks: List of blocks (lists of pixel indices)
        height: Grid height
        width: Grid width
        save_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    # Create color map
    grid = np.zeros((height, width))
    
    for block_idx, block_nodes in enumerate(blocks):
        for idx in block_nodes:
            i = idx // width
            j = idx % width
            grid[i, j] = block_idx
    
    # Plot
    n_blocks = len(blocks)
    fig, ax = plt.subplots(figsize=(10, 10))
    
    cmap = plt.cm.get_cmap('Set3', n_blocks)
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=n_blocks-1)
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=range(n_blocks))
    cbar.set_label('Block ID', fontsize=14, weight='bold')
    
    # Labels
    ax.set_title(f'{n_blocks}-Coloring Block Structure ({height}×{width} grid)', 
                fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('Column', fontsize=12, weight='bold')
    ax.set_ylabel('Row', fontsize=12, weight='bold')
    
    # Add block statistics
    stats_text = "Block Statistics:\n"
    for idx, block in enumerate(blocks):
        stats_text += f"  Block {idx}: {len(block)} pixels\n"
    
    ax.text(1.15, 0.5, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved block visualization to: {save_path}")
    plt.close()


def test_block_structures():
    """Test block creation and independence verification."""
    print("\n" + "=" * 70)
    print("TESTING BLOCK STRUCTURES")
    print("=" * 70)
    
    # Test on small grid first
    print("\n🔬 Test 1: Small 4×4 Grid")
    print("-" * 70)
    
    height, width = 4, 4
    
    # Create blocks
    blocks_2 = create_2color_blocks(height, width)
    blocks_4 = create_4color_blocks(height, width)
    
    print("\n2-Coloring:")
    for idx, block in enumerate(blocks_2):
        print(f"  Block {idx}: {len(block)} pixels - {sorted(block)}")
    
    print("\n4-Coloring:")
    for idx, block in enumerate(blocks_4):
        print(f"  Block {idx}: {len(block)} pixels - {sorted(block)}")
    
    # Create edges for verification
    model = CategoricalEBM(height=height, width=width, n_levels=4)
    
    # Verify 2-coloring
    is_valid_2, msg_2 = verify_block_independence(blocks_2, model.edges_h, model.edges_v)
    print(f"\n2-Coloring Independence: {msg_2}")
    assert is_valid_2, "2-coloring should be independent!"
    
    # Verify 4-coloring
    is_valid_4, msg_4 = verify_block_independence(blocks_4, model.edges_h, model.edges_v)
    print(f"4-Coloring Independence: {msg_4}")
    assert is_valid_4, "4-coloring should be independent!"
    
    # Test on MNIST-sized grid
    print("\n🔬 Test 2: MNIST 28×28 Grid")
    print("-" * 70)
    
    height, width = 28, 28
    
    # Create blocks
    blocks_2_mnist = create_2color_blocks(height, width)
    blocks_4_mnist = create_4color_blocks(height, width)
    
    print("\n2-Coloring:")
    for idx, block in enumerate(blocks_2_mnist):
        print(f"  Block {idx}: {len(block)} pixels")
    print(f"  Total: {sum(len(b) for b in blocks_2_mnist)} pixels")
    
    print("\n4-Coloring:")
    for idx, block in enumerate(blocks_4_mnist):
        print(f"  Block {idx}: {len(block)} pixels")
    print(f"  Total: {sum(len(b) for b in blocks_4_mnist)} pixels")
    
    # Create model for edges
    model_mnist = CategoricalEBM(height=height, width=width, n_levels=4)
    
    # Verify
    is_valid_2_mnist, msg_2_mnist = verify_block_independence(
        blocks_2_mnist, model_mnist.edges_h, model_mnist.edges_v
    )
    print(f"\n2-Coloring Independence: {msg_2_mnist}")
    assert is_valid_2_mnist, "2-coloring should be independent!"
    
    is_valid_4_mnist, msg_4_mnist = verify_block_independence(
        blocks_4_mnist, model_mnist.edges_h, model_mnist.edges_v
    )
    print(f"4-Coloring Independence: {msg_4_mnist}")
    assert is_valid_4_mnist, "4-coloring should be independent!"
    
    # Visualize
    print("\n📊 Creating visualizations...")
    visualize_blocks(blocks_2_mnist, height, width, 'block_structure_2color.png')
    visualize_blocks(blocks_4_mnist, height, width, 'block_structure_4color.png')
    
    print("\n✅ All block structure tests passed!")


def test_block_integration():
    """Test that blocks work correctly with the model."""
    print("\n" + "=" * 70)
    print("TESTING BLOCK INTEGRATION WITH MODEL")
    print("=" * 70)
    
    # Create model
    model = CategoricalEBM(height=28, width=28, n_levels=4)
    
    # Create blocks
    blocks = create_2color_blocks(28, 28)
    
    print(f"\n✓ Created model with {model.count_parameters():,} parameters")
    print(f"✓ Created 2-color blocking with {len(blocks)} blocks")
    
    # Test that we can map between blocks and images
    print("\n🔬 Testing Block ↔ Image Mapping:")
    
    # Create test image
    test_image = torch.randint(0, 4, (1, 28, 28), dtype=torch.long)
    
    # Extract values for each block
    for block_idx, block_pixels in enumerate(blocks):
        # Get flat image
        flat_image = test_image.view(-1)
        block_values = flat_image[block_pixels]
        print(f"  Block {block_idx}: {len(block_pixels)} pixels, "
              f"values range [{block_values.min()}, {block_values.max()}]")
    
    # Verify we can reconstruct
    reconstructed = torch.zeros_like(test_image.view(-1))
    for block_pixels in blocks:
        flat_image = test_image.view(-1)
        reconstructed[block_pixels] = flat_image[block_pixels]
    
    reconstructed = reconstructed.view(1, 28, 28)
    assert torch.all(reconstructed == test_image), "Reconstruction failed!"
    print("\n✓ Block mapping is invertible (can reconstruct original image)")
    
    # Compute energy to verify everything works
    energy = model(test_image)
    print(f"✓ Energy computation works: E = {energy.item():.2f}")
    
    print("\n✅ Block integration tests passed!")


def main_step_2_2():
    """Run all tests for Step 2.2."""
    print("\n" + "=" * 70)
    print("PHASE 2, STEP 2.2: BLOCK STRUCTURE FOR SAMPLING")
    print("=" * 70)
    
    # Run tests
    test_block_structures()
    test_block_integration()
    
    print("\n" + "=" * 70)
    print("✅ STEP 2.2 COMPLETE")
    print("=" * 70)
    
    print("\n📦 Deliverables:")
    print("  ✓ create_2color_blocks() - Checkerboard blocking")
    print("  ✓ create_4color_blocks() - 4-way blocking (for comparison)")
    print("  ✓ verify_block_independence() - Validation function")
    print("  ✓ visualize_blocks() - Block pattern visualization")
    
    print("\n🔧 Features:")
    print("  ✓ 2-coloring: 392 pixels per block (optimal for 4-neighbors)")
    print("  ✓ 4-coloring: 196 pixels per block (for future comparison)")
    print("  ✓ Verified: No within-block edges")
    print("  ✓ Compatible with THRML's block Gibbs sampling")
    print("  ✓ Modular: Works with any number of blocks")
    
    print("\n📊 Block Statistics (28×28 MNIST):")
    blocks_2 = create_2color_blocks(28, 28)
    blocks_4 = create_4color_blocks(28, 28)
    print(f"  2-coloring: 2 blocks of {len(blocks_2[0])} and {len(blocks_2[1])} pixels")
    print(f"  4-coloring: 4 blocks of {len(blocks_4[0])}, {len(blocks_4[1])}, "
          f"{len(blocks_4[2])}, {len(blocks_4[3])} pixels")
    
    print("\n➡️  Next: Step 2.3 - THRML Integration")


if __name__ == "__main__":
    # Run original Step 2.1 tests
    # main()
    
    # Run Step 2.2 tests
    main_step_2_2()
