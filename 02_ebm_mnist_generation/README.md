# Energy-Based Model on MNIST

**Status**: Phase 1 Complete ✅ | Phase 2 In Progress 🚧

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
```

### Run the Project

Follow the implementation plan in `PLAN.md`. Each phase builds on the previous:

1. **Phase 1: Data Preparation** ✅ (Complete)
   - `data_exploration.py` - Analyze MNIST, choose discretization
   - `preprocessing.py` - Discretization functions & data loaders
   - `create_cache.py` - Cache preprocessed data

2. **Phase 2: EBM Architecture** (Next)
   - `ebm_model.py` - Energy function & block structure
   - `thrml_sampler.py` - THRML integration

3. **Phase 3: Training** (Upcoming)
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
├── preprocessed_cache/              # Discretized arrays (gitignored)
│
├── data_exploration.py              # Phase 1.1: Data analysis
├── preprocessing.py                 # Phase 1.2: Preprocessing pipeline
├── create_cache.py                  # Cache creation utility
│
├── *.png                            # Visualizations
└── (more files as project progresses)
```

---

## Key Decisions Made

- **Discretization**: 4 levels {0, 1, 2, 3} - balance quality vs complexity
- **Bins**: [0.0, 0.25, 0.5, 0.75, 1.0]
- **Image size**: 28×28 = 784 variables
- **Train/Val/Test**: 54k / 6k / 10k samples

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

## Next Steps

See `PLAN.md` for the complete 7-phase roadmap. Current status:

- ✅ Phase 1: Data Preparation (COMPLETE)
- 🚧 Phase 2: EBM Architecture Design (IN PROGRESS)
- ⏳ Phase 3-7: Coming soon...

---

## Goal

Answer the question: **Do EBMs + THRML provide speedups over traditional methods for generative modeling?**

We'll measure:
- Sampling speed (samples/second)
- Generation quality (visual inspection, metrics)
- Convergence time (steps to coherent digit)
- Energy efficiency estimates

**Approach**: Honest experimentation with clear documentation of successes AND failures.
