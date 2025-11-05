# Energy-Based Model on MNIST

**Status**: Phase 1 Complete ✅ | Phase 2 Complete ✅ | Phase 3 Next 🚀

Hands-on exploration of Energy-Based Models (EBMs) trained on MNIST and sampled using THRML's block Gibbs sampling. This project validates Extropic's efficiency claims for probabilistic computing on real ML tasks.

---

## Quick Start

### Setup

```bash
# Activate virtual environment (from repo root)
source ../.venv/bin/activate

# Download MNIST and create cache (first time only)
python data_exploration.py      # Step 1.1: Explore data
python preprocessing.py         # Step 1.2: Test preprocessing
python create_cache.py          # Create 418MB preprocessed cache

# Test EBM architecture
python ebm_model.py             # Step 2.1-2.2: Energy function & blocks
python thrml_sampler.py         # Step 2.3: THRML integration
```

### Run the Project

Follow the implementation plan in `PLAN.md`. Each phase builds on the previous:

1. **Phase 1: Data Preparation** ✅ (Complete)
   - `data_exploration.py` - Analyze MNIST, choose discretization
   - `preprocessing.py` - Discretization functions & data loaders
   - `create_cache.py` - Cache preprocessed data

2. **Phase 2: EBM Architecture** ✅ (Complete)
   - `ebm_model.py` - Energy function (Potts model) & block structure (2-coloring)
   - `thrml_sampler.py` - JAX-based Gibbs sampler with block updates

3. **Phase 3: Training** (Next)
   - `train_ebm.py` - Contrastive Divergence training

4. **Phase 4-7**: Sampling, benchmarking, visualization, analysis

---

## Project Structure

```
02_ebm_mnist_generation/
├── PLAN.md                          # Complete implementation roadmap
├── README.md                        # This file
│
├── data/                            # Raw MNIST (gitignored)
├── preprocessed_cache/              # Discretized arrays (418.7 MB, gitignored)
│
├── data_exploration.py              # Phase 1.1: Data analysis ✅
├── preprocessing.py                 # Phase 1.2: Preprocessing pipeline ✅
├── create_cache.py                  # Cache creation utility ✅
│
├── ebm_model.py                     # Phase 2.1-2.2: EBM + blocks ✅
├── thrml_sampler.py                 # Phase 2.3: THRML integration ✅
│
├── *.png                            # Visualizations
└── (more files as project progresses)
```

---

## Key Decisions Made

### Data & Discretization
- **Discretization**: 4 levels {0, 1, 2, 3} - balance quality vs complexity
- **Bins**: [0.0, 0.25, 0.5, 0.75, 1.0]
- **Image size**: 28×28 = 784 variables
- **Train/Val/Test**: 54k / 6k / 10k samples
- **Cache**: 418.7 MB preprocessed discrete arrays

### EBM Architecture
- **Energy Function**: Potts model (categorical variables)
  - E(x; θ) = -Σ bias_i[x_i] - Σ weight * δ(x_i, x_j)
- **Parameters**: 3,138 total
  - 3,136 bias parameters (28×28×4)
  - 2 weight parameters (horizontal + vertical edges)
- **Graph Structure**: 4-neighbor grid (1,512 edges)
- **Block Structure**: 2-coloring (checkerboard)
  - 2 blocks of 392 pixels each
  - Enables parallel Gibbs updates

### THRML Integration
- **Framework**: JAX-based for GPU acceleration
- **Sampler**: Custom block Gibbs implementation
- **Energy Consistency**: JAX vs PyTorch < 0.00003 difference
- **Verified on**: 4×4 and 28×28 grids

---

## Usage Examples

### Load Preprocessed Data

```python
from create_cache import load_preprocessed_mnist

# Fast loading from cache (~instant)
train_images, train_labels, test_images, test_labels, config = \
    load_preprocessed_mnist()

# Images are (N, 28, 28) with discrete values {0, 1, 2, 3}
```

### Use EBM Model

```python
from ebm_model import CategoricalEBM
import torch

# Create model
model = CategoricalEBM(height=28, width=28, n_levels=4)
print(f"Parameters: {model.count_parameters():,}")  # 3,138

# Compute energy
images = torch.randint(0, 4, (batch_size, 28, 28))
energies = model(images)  # (batch_size,)

# Save/load model
model.save('ebm_checkpoint.pt')
model = CategoricalEBM.load('ebm_checkpoint.pt')
```

### Use THRML Sampler

```python
from thrml_sampler import THRMLSampler
from ebm_model import CategoricalEBM

# Create EBM and sampler
ebm = CategoricalEBM(height=28, width=28, n_levels=4)
sampler = THRMLSampler(ebm, n_coloring=2, seed=42)

# Sample single image
sample = sampler.sample(n_steps=100)  # (784,)
image = sampler.state_to_image(sample)  # (28, 28)

# Batch sampling
samples = sampler.sample_batch(batch_size=10, n_steps=100)  # (10, 784)
images = sampler.state_to_image(samples)  # (10, 28, 28)
```

### Use Data Loaders

```python
from preprocessing import create_data_loaders

train_loader, val_loader, test_loader = create_data_loaders(
    batch_size=64,
    val_split=0.1,
    discretize=True
)

# Iterate through batches
for images, labels in train_loader:
    # images: (batch_size, 28, 28) discrete
    # labels: (batch_size,) digit labels
    pass
```

### Discretize/Undiscretize

```python
from preprocessing import discretize_image, undiscretize_image

# Continuous [0,1] → Discrete {0,1,2,3}
discrete = discretize_image(continuous_image)

# Discrete {0,1,2,3} → Continuous [0,1] (for visualization)
continuous = undiscretize_image(discrete)
```

---

## Implementation Status

See `PLAN.md` for the complete 7-phase roadmap. Current status:

- ✅ **Phase 1: Data Preparation** (COMPLETE)
  - Step 1.1: Data exploration with 4-level discretization decision
  - Step 1.2: Preprocessing pipeline with 418.7 MB cache
  
- ✅ **Phase 2: EBM Architecture Design** (COMPLETE)
  - Step 2.1: Potts energy function with 3,138 parameters
  - Step 2.2: 2-coloring block structure (392 pixels/block)
  - Step 2.3: JAX-based THRML sampler with energy consistency verified
  
- 🚀 **Phase 3: Training** (NEXT)
  - Step 3.1: Contrastive Divergence implementation
  - Step 3.2: Training monitoring & logging
  - Step 3.3: Hyperparameter tuning
  
- ⏳ **Phase 4-7**: Sampling, benchmarking, visualization, documentation

---

## Performance Notes

Current implementation prioritizes correctness over speed:
- Energy computation: Vectorized (16.5x faster than naive)
- Gibbs sampling: Python loops (will optimize in Phase 4)
- Future optimizations: JIT compilation, vectorization, GPU acceleration

---

## Goal

Answer the question: **Do EBMs + THRML provide speedups over traditional methods for generative modeling?**

We'll measure:
- Sampling speed (samples/second)
- Generation quality (visual inspection, metrics)
- Convergence time (steps to coherent digit)
- Energy efficiency estimates

**Approach**: Honest experimentation with clear documentation of successes AND failures.
