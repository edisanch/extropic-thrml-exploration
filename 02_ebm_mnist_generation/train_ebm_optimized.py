#!/usr/bin/env python3
"""
EBM Training with Optimized THRML Sampler
==========================================

Phase 4, Step 4.5: Integration of Optimized Sampler

This is the production version of train_ebm_monitored.py that uses the
optimized NativeTHRMLSampler with vectorized batch sampling for proper
CD-1 training with Gibbs negatives.

Key improvements over train_ebm_monitored.py:
- Uses NativeTHRMLSampler instead of THRMLSampler
- Enables use_gibbs_in_training=True (now fast!)
- Vectorized batch sampling (2.4x speedup)
- Can afford more CD steps (cd_steps=5 instead of 1)
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
from thrml_sampler_native import NativeTHRMLSampler  # ← Optimized sampler!
from create_cache import load_preprocessed_mnist
from preprocessing import undiscretize_image


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Configuration for optimized CD-1 training."""
    
    # Data
    target_digit = 3
    batch_size = 64
    
    # Training
    n_epochs = 15  # More epochs now that we're using proper Gibbs
    learning_rate = 1e-3
    cd_steps = 5  # Can afford more steps now! (was 1)
    
    # Optimizer
    optimizer_type = 'adam'
    weight_decay = 0.0
    grad_clip_norm = 1.0
    
    # Model
    image_height = 28
    image_width = 28
    n_levels = 4
    
    # Sampling - NOW ENABLED!
    init_strategy = 'random'
    use_gibbs_in_training = True  # ✅ Now fast enough!
    
    # Monitoring & Logging
    log_dir = './runs'
    log_every_n_steps = 10
    log_samples_every_n_epochs = 1
    n_samples_to_log = 16
    n_gibbs_steps_for_logging = 20  # Can afford real Gibbs for logging now
    log_histograms_every_n_epochs = 1
    
    # Checkpointing
    checkpoint_dir = './checkpoints'
    save_every_n_epochs = 5
    
    # Random seed
    seed = 42


# ============================================================================
# DATA LOADING (unchanged from train_ebm_monitored.py)
# ============================================================================

def load_single_digit_data(digit: int, 
                          batch_size: int,
                          seed: int = 42) -> Tuple[torch.utils.data.DataLoader, int]:
    """Load only samples of a specific digit from MNIST."""
    print("=" * 70)
    print(f"LOADING SINGLE DIGIT DATA (digit={digit})")
    print("=" * 70)
    
    train_images, train_labels, _, _, config = load_preprocessed_mnist()
    
    mask = (train_labels == digit)
    digit_images = train_images[mask]
    digit_labels = train_labels[mask]
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total MNIST training samples: {len(train_images):,}")
    print(f"  Samples of digit {digit}: {len(digit_images):,}")
    print(f"  Percentage: {100 * len(digit_images) / len(train_images):.1f}%")
    
    images_tensor = torch.from_numpy(digit_images).long()
    labels_tensor = torch.from_numpy(digit_labels).long()
    
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
# TENSORBOARD LOGGING (unchanged)
# ============================================================================

def log_images_to_tensorboard(writer: SummaryWriter,
                              images: np.ndarray,
                              tag: str,
                              global_step: int,
                              n_levels: int = 4):
    """Log images to TensorBoard."""
    images_continuous = np.array([
        undiscretize_image(img, n_levels=n_levels) 
        for img in images
    ])
    
    images_torch = torch.from_numpy(images_continuous).float().unsqueeze(1)
    
    grid = torchvision.utils.make_grid(
        images_torch,
        nrow=4,
        normalize=False,
        scale_each=False,
        pad_value=1.0
    )
    
    writer.add_image(tag, grid, global_step)


def log_model_parameters(writer: SummaryWriter,
                         ebm: CategoricalEBM,
                         global_step: int):
    """Log model parameter distributions."""
    writer.add_histogram('parameters/biases', ebm.biases, global_step)
    writer.add_scalar('parameters/weight_h', ebm.weight_h.item(), global_step)
    writer.add_scalar('parameters/weight_v', ebm.weight_v.item(), global_step)
    
    summary = ebm.get_parameter_summary()
    for key, val in summary.items():
        writer.add_scalar(f'parameters/{key}', val, global_step)


# ============================================================================
# OPTIMIZED SAMPLING FOR LOGGING
# ============================================================================

def generate_samples_for_logging(ebm: CategoricalEBM,
                                 sampler: NativeTHRMLSampler,
                                 n_samples: int,
                                 n_gibbs_steps: int,
                                 height: int,
                                 width: int) -> np.ndarray:
    """
    Generate samples for logging using optimized batch sampling.
    
    Returns:
        samples: (n_samples, height, width) numpy array
    """
    if n_gibbs_steps == 0:
        # Random samples
        key = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
        samples = jax.random.randint(key, (n_samples, height * width), 0, 4)
        samples = np.array(samples).reshape(n_samples, height, width)
        return samples
    
    # Use vectorized batch sampling! (Fast!)
    samples_flat = sampler.sample_batch_vmap(
        batch_size=n_samples,
        n_steps=n_gibbs_steps
    )
    
    samples = np.array(samples_flat).reshape(n_samples, height, width)
    return samples


# ============================================================================
# OPTIMIZED CD-1 TRAINING STEP
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


def cd1_step_optimized(ebm: CategoricalEBM,
                      sampler: NativeTHRMLSampler,
                      data_batch: torch.Tensor,
                      optimizer: torch.optim.Optimizer,
                      config: TrainingConfig) -> dict:
    """
    Perform one CD-1 training step with OPTIMIZED Gibbs sampling.
    
    Key improvement: Uses vectorized batch_vmap sampling instead of
    sequential sampling, providing 2.4x speedup.
    
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
    # NEGATIVE PHASE (OPTIMIZED!)
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
        # ✅ OPTIMIZED: Use vectorized batch sampling (2.4x faster!)
        samples_jax = sampler.sample_batch_vmap(
            batch_size=batch_size,
            n_steps=config.cd_steps,
            initial_states=initial_states
        )
    else:
        # Fallback: Use random samples (not recommended)
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
# TRAINING LOOP
# ============================================================================

def train_ebm_optimized(config: TrainingConfig):
    """
    Train EBM using OPTIMIZED CD-1 with proper Gibbs sampling.
    """
    print("\n" + "=" * 70)
    print("PHASE 4, STEP 4.5: OPTIMIZED CD-1 TRAINING")
    print("=" * 70)
    print("\n🚀 Using NativeTHRMLSampler with vectorized batch sampling!")
    print(f"   use_gibbs_in_training = {config.use_gibbs_in_training}")
    print(f"   cd_steps = {config.cd_steps}")
    
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create directories
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_subdir = f"digit{config.target_digit}_optimized_{timestamp}"
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
    # CREATE OPTIMIZED SAMPLER
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("INITIALIZING OPTIMIZED THRML SAMPLER")
    print("=" * 70)
    
    sampler = NativeTHRMLSampler(
        ebm=ebm,
        seed=config.seed
    )
    
    print("\n✓ Using sample_batch_vmap() for 2.4x faster sampling!")
    
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
    config_text += f"CD steps: {config.cd_steps} (improved from 1!)\n"
    config_text += f"Optimizer: {config.optimizer_type}\n"
    config_text += f"Grad clip: {config.grad_clip_norm}\n"
    config_text += f"Use Gibbs in training: {config.use_gibbs_in_training} ✅\n"
    config_text += f"Sampler: NativeTHRMLSampler (optimized)\n"
    
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
    print(f"  Use Gibbs: {config.use_gibbs_in_training} ✅")
    print(f"  CD steps: {config.cd_steps}")
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
            # CD-1 training step (OPTIMIZED!)
            metrics = cd1_step_optimized(
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
                writer.add_histogram('distributions/energy_data', 
                                   metrics['energy_data_tensor'], global_step)
                writer.add_histogram('distributions/energy_samples',
                                   metrics['energy_samples_tensor'], global_step)
            
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
        print(f"  Energy gap: {avg_epoch_energy_data - avg_epoch_energy_samples:.2f}")
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
                width=config.image_width
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
            checkpoint_path = Path(config.checkpoint_dir) / f"ebm_digit{config.target_digit}_optimized_epoch{epoch}.pt"
            ebm.save(str(checkpoint_path))
            print(f"  💾 Saved checkpoint: {checkpoint_path}")
        
        print(f"{'='*70}\n")
    
    # ========================================================================
    # SAVE FINAL MODEL
    # ========================================================================
    
    final_model_path = Path(config.checkpoint_dir) / f"ebm_digit{config.target_digit}_optimized_final.pt"
    ebm.save(str(final_model_path))
    
    writer.close()
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n📦 Final model: {final_model_path}")
    print(f"📊 TensorBoard logs: {Path(config.log_dir) / log_subdir}")
    print(f"\n🚀 View results: tensorboard --logdir={config.log_dir}")
    
    print("\n✨ Training improvements:")
    print(f"   ✅ Using proper Gibbs sampling (not random negatives)")
    print(f"   ✅ Vectorized batch sampling (2.4x faster)")
    print(f"   ✅ CD-{config.cd_steps} instead of CD-1")
    print(f"   ✅ Native THRML sampler (0.27ms per Gibbs step)")
    
    return ebm, sampler


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run optimized CD-1 training."""
    print("\n" + "=" * 70)
    print("PHASE 4, STEP 4.5: OPTIMIZED TRAINING WITH NATIVE SAMPLER")
    print("=" * 70)
    
    config = TrainingConfig()
    ebm, sampler = train_ebm_optimized(config)
    
    print("\n" + "=" * 70)
    print("✅ STEP 4.5 COMPLETE")
    print("=" * 70)
    
    print("\n📦 Deliverables:")
    print("  ✓ Integrated NativeTHRMLSampler into training")
    print("  ✓ Enabled use_gibbs_in_training=True")
    print("  ✓ Training with proper Gibbs negatives (not random)")
    print("  ✓ Vectorized batch sampling (2.4x speedup)")
    print("  ✓ CD-5 training (improved from CD-1)")
    print("  ✓ Full TensorBoard monitoring")
    
    print("\n➡️  Next: Phase 5 - Hyperparameter tuning with optimized sampler")


if __name__ == "__main__":
    main()
