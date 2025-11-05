#!/usr/bin/env python3
"""
EBM Training with TensorBoard Monitoring
=========================================

Phase 3, Step 3.2: Training Monitoring & Visualization

Implements CD-1 training with comprehensive TensorBoard logging:
- Scalar metrics: loss, energies, gradients, learning rates
- Histograms: energy distributions, parameter distributions
- Images: generated samples, real samples, energy heatmaps
- Text: configuration, model architecture

Run TensorBoard with: tensorboard --logdir=runs
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import time
from typing import Tuple, Optional
from datetime import datetime

# TensorBoard
from torch.utils.tensorboard import SummaryWriter
import torchvision

from ebm_model import CategoricalEBM
from thrml_sampler import THRMLSampler
from create_cache import load_preprocessed_mnist
from preprocessing import undiscretize_image


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Configuration for CD-1 training with monitoring."""
    
    # Data
    target_digit = 3  # Train on digit 3
    batch_size = 64
    
    # Training
    n_epochs = 10  # Full training
    learning_rate = 1e-3
    cd_steps = 1  # CD-1
    
    # Optimizer
    optimizer_type = 'adam'
    weight_decay = 0.0
    grad_clip_norm = 1.0
    
    # Model
    image_height = 28
    image_width = 28
    n_levels = 4
    
    # Sampling
    init_strategy = 'random'
    use_gibbs_in_training = False  # Set True when Phase 4 optimization done
    
    # Monitoring & Logging
    log_dir = './runs'
    log_every_n_steps = 10  # Log scalars every N steps
    log_samples_every_n_epochs = 1  # Generate and log samples
    n_samples_to_log = 16  # Number of samples to generate
    n_gibbs_steps_for_logging = 0  # Set to 0 to skip Gibbs (use random) - fast!
    log_histograms_every_n_epochs = 1  # Log parameter histograms
    
    # Checkpointing
    checkpoint_dir = './checkpoints'
    save_every_n_epochs = 5
    
    # Random seed
    seed = 42


# ============================================================================
# DATA LOADING
# ============================================================================

def load_single_digit_data(digit: int, 
                          batch_size: int,
                          seed: int = 42) -> Tuple[torch.utils.data.DataLoader, int]:
    """Load only samples of a specific digit from MNIST."""
    print("=" * 70)
    print(f"LOADING SINGLE DIGIT DATA (digit={digit})")
    print("=" * 70)
    
    # Load preprocessed MNIST
    train_images, train_labels, _, _, config = load_preprocessed_mnist()
    
    # Filter for target digit
    mask = (train_labels == digit)
    digit_images = train_images[mask]
    digit_labels = train_labels[mask]
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total MNIST training samples: {len(train_images):,}")
    print(f"  Samples of digit {digit}: {len(digit_images):,}")
    print(f"  Percentage: {100 * len(digit_images) / len(train_images):.1f}%")
    
    # Convert to torch tensors
    images_tensor = torch.from_numpy(digit_images).long()
    labels_tensor = torch.from_numpy(digit_labels).long()
    
    # Create dataset and loader
    dataset = torch.utils.data.TensorDataset(images_tensor, labels_tensor)
    generator = torch.Generator().manual_seed(seed)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=True
    )
    
    print(f"\n✓ Data loader created:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {len(data_loader)}")
    
    return data_loader, len(dataset)


# ============================================================================
# TENSORBOARD LOGGING UTILITIES
# ============================================================================

def log_images_to_tensorboard(writer: SummaryWriter,
                              images: np.ndarray,
                              tag: str,
                              global_step: int,
                              n_levels: int = 4):
    """
    Log images to TensorBoard.
    
    Args:
        writer: TensorBoard writer
        images: Images array (batch, H, W) with discrete values
        tag: Tag for the images
        global_step: Global training step
        n_levels: Number of discrete levels
    """
    # Undiscretize for visualization
    images_continuous = np.array([
        undiscretize_image(img, n_levels=n_levels) 
        for img in images
    ])
    
    # Convert to torch tensor (B, 1, H, W)
    images_torch = torch.from_numpy(images_continuous).float().unsqueeze(1)
    
    # Create grid
    grid = torchvision.utils.make_grid(
        images_torch,
        nrow=4,
        normalize=False,
        scale_each=False,
        pad_value=1.0
    )
    
    writer.add_image(tag, grid, global_step)


def log_energy_histogram(writer: SummaryWriter,
                         energies: torch.Tensor,
                         tag: str,
                         global_step: int):
    """Log energy distribution as histogram."""
    writer.add_histogram(tag, energies, global_step)


def log_model_parameters(writer: SummaryWriter,
                         ebm: CategoricalEBM,
                         global_step: int):
    """Log model parameter distributions."""
    # Biases
    writer.add_histogram('parameters/biases', ebm.biases, global_step)
    
    # Weights
    writer.add_scalar('parameters/weight_h', ebm.weight_h.item(), global_step)
    writer.add_scalar('parameters/weight_v', ebm.weight_v.item(), global_step)
    
    # Summary statistics
    summary = ebm.get_parameter_summary()
    for key, val in summary.items():
        writer.add_scalar(f'parameters/{key}', val, global_step)


# ============================================================================
# SAMPLING FOR VISUALIZATION
# ============================================================================

def generate_samples_for_logging(ebm: CategoricalEBM,
                                 sampler: THRMLSampler,
                                 n_samples: int,
                                 n_gibbs_steps: int,
                                 height: int,
                                 width: int,
                                 n_levels: int) -> np.ndarray:
    """
    Generate samples for logging (using few Gibbs steps for speed).
    
    Returns:
        samples: (n_samples, height, width) numpy array
    """
    if n_gibbs_steps == 0:
        # Just return random samples (no Gibbs)
        key = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
        samples = jax.random.randint(
            key, 
            (n_samples, height * width), 
            0, 
            n_levels
        )
        samples = np.array(samples).reshape(n_samples, height, width)
        return samples
    
    # Generate with Gibbs sampling
    samples = []
    for i in range(n_samples):
        sample = sampler.sample(n_steps=n_gibbs_steps)
        samples.append(sample)
    
    samples = np.array(samples).reshape(n_samples, height, width)
    return samples


# ============================================================================
# CD-1 TRAINING WITH MONITORING
# ============================================================================

def initialize_negative_samples(batch_size: int,
                               height: int,
                               width: int,
                               n_levels: int,
                               strategy: str = 'random') -> jnp.ndarray:
    """Initialize negative samples for CD."""
    n_pixels = height * width
    
    if strategy == 'random':
        key = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
        states = jax.random.randint(key, (batch_size, n_pixels), 0, n_levels)
        return states
    else:
        raise ValueError(f"Unknown initialization strategy: {strategy}")


def cd1_step(ebm: CategoricalEBM,
             sampler: THRMLSampler,
             data_batch: torch.Tensor,
             optimizer: torch.optim.Optimizer,
             config: TrainingConfig) -> dict:
    """
    Perform one CD-1 training step with detailed metrics.
    
    Returns:
        metrics: Dictionary with loss, energies, gradients, etc.
    """
    batch_size = data_batch.shape[0]
    
    # ========================================================================
    # POSITIVE PHASE
    # ========================================================================
    
    optimizer.zero_grad()
    energy_data = ebm(data_batch)
    loss_positive = energy_data.mean()
    loss_positive.backward()
    
    positive_grads = {}
    for name, param in ebm.named_parameters():
        if param.grad is not None:
            positive_grads[name] = param.grad.clone()
    
    # ========================================================================
    # NEGATIVE PHASE
    # ========================================================================
    
    # Initialize negative samples
    initial_states = initialize_negative_samples(
        batch_size=batch_size,
        height=config.image_height,
        width=config.image_width,
        n_levels=config.n_levels,
        strategy=config.init_strategy
    )
    
    if config.use_gibbs_in_training:
        # Run Gibbs sampling (slow - will optimize in Phase 4)
        samples_list = []
        for i in range(batch_size):
            sample = sampler.sample(
                n_steps=config.cd_steps,
                initial_state=initial_states[i]
            )
            samples_list.append(sample)
        samples_jax = jnp.stack(samples_list)
    else:
        # Use random samples (fast but less accurate)
        samples_jax = initial_states
    
    # Convert to torch
    samples_torch = torch.from_numpy(np.array(samples_jax)).long()
    samples_torch = samples_torch.view(batch_size, config.image_height, config.image_width)
    
    # Compute energy on samples
    with torch.no_grad():
        energy_samples = ebm(samples_torch)
    
    optimizer.zero_grad()
    energy_samples_grad = ebm(samples_torch)
    loss_negative = energy_samples_grad.mean()
    loss_negative.backward()
    
    negative_grads = {}
    for name, param in ebm.named_parameters():
        if param.grad is not None:
            negative_grads[name] = param.grad.clone()
    
    # ========================================================================
    # CONTRASTIVE DIVERGENCE UPDATE
    # ========================================================================
    
    # Set gradients
    for name, param in ebm.named_parameters():
        if param.grad is not None:
            param.grad = positive_grads[name] - negative_grads[name]
    
    # Gradient clipping
    if config.grad_clip_norm > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            ebm.parameters(),
            config.grad_clip_norm
        )
    else:
        grad_norm = sum(p.grad.norm().item() ** 2 for p in ebm.parameters()) ** 0.5
    
    # Update parameters
    optimizer.step()
    
    # ========================================================================
    # COLLECT METRICS
    # ========================================================================
    
    metrics = {
        'loss': loss_positive.item() - loss_negative.item(),
        'energy_data': energy_data.mean().item(),
        'energy_samples': energy_samples.mean().item(),
        'energy_gap': energy_data.mean().item() - energy_samples.mean().item(),
        'grad_norm': grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
        'energy_data_std': energy_data.std().item(),
        'energy_samples_std': energy_samples.std().item(),
        'energy_data_tensor': energy_data,
        'energy_samples_tensor': energy_samples,
    }
    
    return metrics


# ============================================================================
# TRAINING LOOP WITH TENSORBOARD
# ============================================================================

def train_ebm_monitored(config: TrainingConfig):
    """
    Train EBM using CD-1 with comprehensive TensorBoard monitoring.
    """
    print("\n" + "=" * 70)
    print("PHASE 3, STEP 3.2: CD-1 TRAINING WITH MONITORING")
    print("=" * 70)
    
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create directories
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create TensorBoard writer with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_subdir = f"digit{config.target_digit}_{timestamp}"
    writer = SummaryWriter(log_dir=Path(config.log_dir) / log_subdir)
    
    print(f"\n📊 TensorBoard logging to: {Path(config.log_dir) / log_subdir}")
    print(f"   Run: tensorboard --logdir={config.log_dir}")
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    
    train_loader, n_samples = load_single_digit_data(
        digit=config.target_digit,
        batch_size=config.batch_size,
        seed=config.seed
    )
    
    # Log some real data samples
    first_batch = next(iter(train_loader))[0]
    log_images_to_tensorboard(
        writer,
        first_batch[:16].numpy(),
        'data/real_samples',
        0,
        config.n_levels
    )
    
    # ========================================================================
    # CREATE MODEL
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("INITIALIZING MODEL")
    print("=" * 70)
    
    ebm = CategoricalEBM(
        height=config.image_height,
        width=config.image_width,
        n_levels=config.n_levels
    )
    
    # Log model architecture
    summary = ebm.get_parameter_summary()
    model_text = "Model Architecture\n" + "="*50 + "\n"
    model_text += f"Image size: {config.image_height}×{config.image_width}\n"
    model_text += f"Discrete levels: {config.n_levels}\n"
    model_text += f"Total parameters: {sum(p.numel() for p in ebm.parameters())}\n\n"
    model_text += "Initial Parameters:\n"
    for key, val in summary.items():
        model_text += f"  {key}: {val:.6f}\n"
    
    writer.add_text('model/architecture', model_text, 0)
    
    # ========================================================================
    # CREATE SAMPLER
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("INITIALIZING THRML SAMPLER")
    print("=" * 70)
    
    sampler = THRMLSampler(
        ebm=ebm,
        n_coloring=2,
        seed=config.seed
    )
    
    # ========================================================================
    # CREATE OPTIMIZER
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("INITIALIZING OPTIMIZER")
    print("=" * 70)
    
    if config.optimizer_type == 'adam':
        optimizer = optim.Adam(
            ebm.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer_type}")
    
    # Log configuration
    config_text = "Training Configuration\n" + "="*50 + "\n"
    config_text += f"Target digit: {config.target_digit}\n"
    config_text += f"Epochs: {config.n_epochs}\n"
    config_text += f"Batch size: {config.batch_size}\n"
    config_text += f"Learning rate: {config.learning_rate}\n"
    config_text += f"CD steps: {config.cd_steps}\n"
    config_text += f"Optimizer: {config.optimizer_type}\n"
    config_text += f"Grad clip: {config.grad_clip_norm}\n"
    config_text += f"Use Gibbs in training: {config.use_gibbs_in_training}\n"
    
    writer.add_text('config/training', config_text, 0)
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    print(f"\n⚙️  Configuration:")
    print(f"  Target digit: {config.target_digit}")
    print(f"  Epochs: {config.n_epochs}")
    print(f"  Batches per epoch: {len(train_loader)}")
    print(f"  Use Gibbs: {config.use_gibbs_in_training}")
    print(f"  Log scalars every: {config.log_every_n_steps} steps")
    print(f"  Log samples every: {config.log_samples_every_n_epochs} epochs")
    
    global_step = 0
    
    for epoch in range(1, config.n_epochs + 1):
        epoch_start_time = time.time()
        
        # Accumulators
        epoch_loss = 0.0
        epoch_energy_data = 0.0
        epoch_energy_samples = 0.0
        epoch_grad_norm = 0.0
        
        for batch_idx, (data_batch, labels) in enumerate(train_loader):
            # CD-1 training step
            metrics = cd1_step(
                ebm=ebm,
                sampler=sampler,
                data_batch=data_batch,
                optimizer=optimizer,
                config=config
            )
            
            # Accumulate
            epoch_loss += metrics['loss']
            epoch_energy_data += metrics['energy_data']
            epoch_energy_samples += metrics['energy_samples']
            epoch_grad_norm += metrics['grad_norm']
            
            global_step += 1
            
            # Log scalars
            if global_step % config.log_every_n_steps == 0:
                writer.add_scalar('train/loss', metrics['loss'], global_step)
                writer.add_scalar('train/energy_data', metrics['energy_data'], global_step)
                writer.add_scalar('train/energy_samples', metrics['energy_samples'], global_step)
                writer.add_scalar('train/energy_gap', metrics['energy_gap'], global_step)
                writer.add_scalar('train/grad_norm', metrics['grad_norm'], global_step)
                writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step)
                
                # Log energy distributions
                log_energy_histogram(
                    writer,
                    metrics['energy_data_tensor'],
                    'distributions/energy_data',
                    global_step
                )
                log_energy_histogram(
                    writer,
                    metrics['energy_samples_tensor'],
                    'distributions/energy_samples',
                    global_step
                )
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                avg_e_data = epoch_energy_data / (batch_idx + 1)
                avg_e_samples = epoch_energy_samples / (batch_idx + 1)
                
                print(f"Epoch {epoch}/{config.n_epochs} | "
                      f"Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {avg_loss:7.2f} | "
                      f"E_data: {avg_e_data:7.2f} | "
                      f"E_samp: {avg_e_samples:7.2f}")
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        n_batches = len(train_loader)
        
        avg_epoch_loss = epoch_loss / n_batches
        avg_epoch_energy_data = epoch_energy_data / n_batches
        avg_epoch_energy_samples = epoch_energy_samples / n_batches
        avg_epoch_grad_norm = epoch_grad_norm / n_batches
        
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch} SUMMARY")
        print(f"{'='*70}")
        print(f"  Loss: {avg_epoch_loss:.2f}")
        print(f"  Energy (data): {avg_epoch_energy_data:.2f}")
        print(f"  Energy (samples): {avg_epoch_energy_samples:.2f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Log epoch summaries
        writer.add_scalar('epoch/loss', avg_epoch_loss, epoch)
        writer.add_scalar('epoch/energy_data', avg_epoch_energy_data, epoch)
        writer.add_scalar('epoch/energy_samples', avg_epoch_energy_samples, epoch)
        writer.add_scalar('epoch/time', epoch_time, epoch)
        
        # Generate and log samples
        if epoch % config.log_samples_every_n_epochs == 0:
            print(f"\n  Generating {config.n_samples_to_log} samples for logging...")
            samples = generate_samples_for_logging(
                ebm=ebm,
                sampler=sampler,
                n_samples=config.n_samples_to_log,
                n_gibbs_steps=config.n_gibbs_steps_for_logging,
                height=config.image_height,
                width=config.image_width,
                n_levels=config.n_levels
            )
            
            log_images_to_tensorboard(
                writer,
                samples,
                f'samples/generated_epoch_{epoch}',
                global_step,
                config.n_levels
            )
            print(f"  ✓ Logged samples to TensorBoard")
        
        # Log parameter histograms
        if epoch % config.log_histograms_every_n_epochs == 0:
            log_model_parameters(writer, ebm, global_step)
        
        # Save checkpoint
        if epoch % config.save_every_n_epochs == 0:
            checkpoint_path = Path(config.checkpoint_dir) / f"ebm_digit{config.target_digit}_epoch{epoch}.pt"
            ebm.save(str(checkpoint_path))
            print(f"  💾 Saved checkpoint: {checkpoint_path}")
        
        print(f"{'='*70}\n")
    
    # ========================================================================
    # SAVE FINAL MODEL
    # ========================================================================
    
    final_model_path = Path(config.checkpoint_dir) / f"ebm_digit{config.target_digit}_monitored_final.pt"
    ebm.save(str(final_model_path))
    
    writer.close()
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n📦 Final model: {final_model_path}")
    print(f"📊 TensorBoard logs: {Path(config.log_dir) / log_subdir}")
    print(f"\n🚀 View results: tensorboard --logdir={config.log_dir}")
    
    return ebm, sampler


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run CD-1 training with TensorBoard monitoring."""
    print("\n" + "=" * 70)
    print("PHASE 3, STEP 3.2: TRAINING WITH MONITORING")
    print("=" * 70)
    
    config = TrainingConfig()
    ebm, sampler = train_ebm_monitored(config)
    
    print("\n" + "=" * 70)
    print("✅ STEP 3.2 COMPLETE")
    print("=" * 70)
    
    print("\n📦 Deliverables:")
    print("  ✓ TensorBoard integration")
    print("  ✓ Scalar logging (loss, energies, gradients)")
    print("  ✓ Histogram logging (energy distributions, parameters)")
    print("  ✓ Image logging (real data, generated samples)")
    print("  ✓ Text logging (config, architecture)")
    print("  ✓ Per-epoch sample generation")
    print("  ✓ Comprehensive metrics tracking")
    
    print("\n➡️  Next: Step 3.3 - Advanced Training Techniques")
    print("     (Learning rate scheduling, early stopping, etc.)")


if __name__ == "__main__":
    main()
