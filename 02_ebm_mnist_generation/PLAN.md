# Energy-Based Model on MNIST: Implementation Plan

## Project Goal
Build an Energy-Based Model (EBM) that generates MNIST digit images using THRML's block Gibbs sampling. Validate Extropic's efficiency claims for real ML tasks.

---

## Phase 1: Data Preparation & Understanding ✅ COMPLETE

### Step 1.1: Download & Explore MNIST ✅
**File**: `data_exploration.py`
- ✅ Download MNIST dataset (60k train, 10k test)
- ✅ Visualize sample digits
- ✅ Analyze pixel value distributions
- ✅ Decide on discretization strategy (binary, 4-level, or 16-level)

**Deliverable**: Understanding of data characteristics, discretization choice
**Decision**: 4-level discretization chosen for balance of detail and complexity

### Step 1.2: Preprocessing Pipeline ✅
**File**: `preprocessing.py` + `create_cache.py`
- ✅ Implement discretization function (continuous → discrete pixels)
- ✅ Create train/validation splits
- ✅ Prepare data loaders for training
- ✅ Cache preprocessed data (418.7 MB cache)

**Deliverable**: Clean, discretized MNIST ready for EBM training

---

## Phase 2: EBM Architecture Design ✅ COMPLETE

### Step 2.1: Define Energy Function ✅
**File**: `ebm_model.py`
- ✅ Design local interaction structure (4-neighbors like Ising)
- ✅ Define learnable parameters:
  - Biases: per-pixel bias terms (3,136 parameters)
  - Weights: pairwise interaction strengths (2 parameters: horizontal + vertical)
- ✅ Implement energy computation: E(x; θ) = -Σ biases·x - Σ weights·x_i·x_j
- ✅ Vectorized fast implementation (16.5x speedup)

**Deliverable**: EBM class with energy function (3,138 total parameters)

### Step 2.2: Block Structure for Sampling ✅
**File**: `ebm_model.py` (continued)
- ✅ Design checkerboard coloring (2-color or 4-color)
- ✅ Map MNIST 28×28 grid to THRML nodes
- ✅ Define independent block updates
- ✅ Verify no within-block interactions
- ✅ Visualize block patterns

**Deliverable**: Block structure compatible with THRML (2-coloring: 392 pixels/block)

### Step 2.3: THRML Integration ✅
**File**: `thrml_sampler.py`
- ✅ Convert EBM to JAX format for sampling
- ✅ Create sampling program with blocks
- ✅ Implement conditional distributions for Gibbs updates
- ✅ Test single-step sampling
- ✅ Verify energy consistency (JAX vs PyTorch < 0.00003 difference)

**Deliverable**: Working THRML sampler for our EBM

---

## Phase 3: Training the EBM

### Step 3.1: Training Algorithm Implementation
**File**: `train_ebm.py`
- Implement Contrastive Divergence (CD-k) or Persistent CD
- Positive phase: compute gradient on real data
- Negative phase: sample from model, compute gradient
- Parameter updates with optimizer (Adam/SGD)

**Choices**:
- CD-1 vs CD-10 (k steps between gradient updates)
- Learning rate schedule
- Batch size (balance speed vs stability)

**Deliverable**: Training loop with gradient computation

### Step 3.2: Training Monitoring
**File**: `train_ebm.py` (continued)
- Log energy values (real data vs samples)
- Track reconstruction error
- Visualize samples during training
- Save checkpoints every N epochs

**Deliverable**: Training with TensorBoard/logging

### Step 3.3: Hyperparameter Tuning
- Experiment with:
  - Learning rates (1e-3, 1e-4, 1e-5)
  - CD steps (1, 5, 10)
  - Batch sizes (32, 64, 128)
  - Initialization strategies
- Select best configuration

**Deliverable**: Trained EBM with tuned hyperparameters

---

## Phase 4: Sampling & Generation

### Step 4.1: THRML Block Gibbs Sampling
**File**: `sample_thrml.py`
- Initialize from random noise
- Run block Gibbs with warmup + collection phases
- Vary sampling schedules (warmup steps, samples between)
- Generate large batches of samples

**Deliverable**: Generated digit samples via THRML

### Step 4.2: Naive Baseline Sampler
**File**: `sample_naive.py`
- Implement sequential single-pixel Gibbs
- No blocking, pure Python loops
- Same model parameters as THRML version

**Deliverable**: Baseline for speed comparison

### Step 4.3: Quality Assessment
**File**: `evaluate_quality.py`
- Visual inspection (grid of samples)
- Per-digit class sampling (can we get all 0-9?)
- Diversity metrics (are samples varied?)
- Human evaluation protocol

**Deliverable**: Quality analysis of generated digits

---

## Phase 5: Benchmarking & Comparison

### Step 5.1: Speed Benchmarks
**File**: `benchmark_sampling.py`
- Measure samples/second for:
  - THRML block Gibbs (GPU)
  - THRML block Gibbs (CPU)
  - Naive sequential Gibbs
- Vary problem sizes (14×14, 28×28, 56×56 if we upscale)
- Find crossover points

**Deliverable**: Sampling speed comparison charts

### Step 5.2: VAE Baseline (Optional)
**File**: `baseline_vae.py`
- Train simple VAE on MNIST
- Measure:
  - Training time
  - Sampling speed
  - Sample quality
  - Energy consumption (if measurable)

**Deliverable**: VAE comparison for context

### Step 5.3: Convergence Analysis
**File**: `convergence_analysis.py`
- Measure steps to coherent digit:
  - Energy trajectory (noise → low energy)
  - Visual coherence (human can recognize digit)
  - Autocorrelation time
- Compare THRML vs naive convergence

**Deliverable**: Convergence metrics and plots

### Step 5.4: Energy Efficiency Estimate
**File**: `energy_analysis.py`
- Estimate compute operations:
  - FLOPs per Gibbs step
  - Memory transfers
- Compare to GPU inference energy
- Project to hypothetical TSU performance
- Reality-check the 10,000x claim

**Deliverable**: Energy efficiency analysis report

---

## Phase 6: Visualization & Analysis

### Step 6.1: Energy Landscape Explorer
**File**: `visualize_energy.py`
- Sample grid of images
- Plot energy values as heatmap
- Visualize sampling trajectories (PCA/t-SNE)
- Show basins of attraction for different digits

**Deliverable**: Energy landscape visualizations

### Step 6.2: Sampling Dynamics Animation
**File**: `animate_sampling.py`
- Record Gibbs sampling steps
- Create animation: noise → digit
- Show block updates in parallel
- Visualize energy decreasing over time

**Deliverable**: Animated sampling process

### Step 6.3: Block Strategy Comparison
**File**: `compare_blocking.py`
- Compare 2-coloring vs 4-coloring
- Measure convergence speed
- Visualize which performs better
- Understand when more blocks help

**Deliverable**: Block strategy analysis

---

## Phase 7: Documentation & Findings

### Step 7.1: Results Summary
**File**: `RESULTS.md`
- Summarize all findings
- Include plots and metrics
- Compare to original claims
- Honest assessment of where EBMs succeed/fail

**Deliverable**: Comprehensive results document

### Step 7.2: Code Documentation
- Add docstrings to all functions
- Create README for directory
- Usage examples for each script
- Requirements and setup instructions

**Deliverable**: Well-documented codebase

### Step 7.3: Update Main README
**File**: `../README.md`
- Mark project as complete
- Add key findings summary
- Link to results and visualizations
- Update benchmark table

**Deliverable**: Updated project README

---

## Implementation Order (Recommended)

### Week 1: Foundation
1. ✅ Setup environment and dependencies
2. **Day 1-2**: Phase 1 (Data preparation)
3. **Day 3-4**: Phase 2.1-2.2 (EBM design)
4. **Day 5-7**: Phase 2.3 (THRML integration)

### Week 2: Training
5. **Day 8-10**: Phase 3.1-3.2 (Training implementation)
6. **Day 11-14**: Phase 3.3 (Hyperparameter tuning + training)

### Week 3: Sampling & Benchmarking
7. **Day 15-16**: Phase 4.1-4.2 (Sampling implementations)
8. **Day 17-18**: Phase 4.3 (Quality assessment)
9. **Day 19-21**: Phase 5 (All benchmarks)

### Week 4: Polish & Documentation
10. **Day 22-24**: Phase 6 (Visualizations)
11. **Day 25-28**: Phase 7 (Documentation)

---

## Key Decision Points

### Decision 1: Discretization Level (Step 1.1)
- **Binary** (0/1): Simplest, like Ising, but loses detail
- **4-level** (0/1/2/3): More expressive, manageable
- **16-level** (0-15): Rich, but sampling complexity increases

**Recommendation**: Start with 4-level, can try binary if needed

### Decision 2: Energy Function Complexity (Step 2.1)
- **Simple**: Only nearest-neighbor interactions (4-neighbors)
- **Moderate**: Add diagonal neighbors (8-neighbors)
- **Complex**: Add longer-range interactions

**Recommendation**: Start simple (4-neighbors)

### Decision 3: Training Algorithm (Step 3.1)
- **CD-1**: Fast but biased gradients
- **CD-10**: Better gradients, slower training
- **Persistent CD**: Most accurate, complex to implement

**Recommendation**: Start with CD-1, upgrade if needed

### Decision 4: Baseline Scope (Step 5.2)
- **Skip VAE**: Focus on sampling speed only
- **Simple VAE**: Basic comparison for context
- **Full comparison**: VAE + GAN + Diffusion (too much scope)

**Recommendation**: Simple VAE for context if time permits

---

## Success Criteria

### Minimum Viable Product (MVP)
- ✅ EBM trains on MNIST (energy decreases)
- ✅ THRML generates recognizable digits
- ✅ Speed benchmark shows crossover point
- ✅ Basic visualizations of samples

### Ideal Outcomes
- Digits are diverse and high quality
- THRML shows 10x+ speedup at scale
- Clear understanding of when EBMs/TSUs make sense
- Reproducible claims validation

### Stretch Goals
- FID score comparable to simple baselines
- Energy landscape exploration reveals structure
- Animation of sampling is publication-quality
- Findings contribute to EBM research community

---

## Risk Mitigation

### Risk 1: EBM Training Fails to Converge
**Mitigation**: 
- Use simpler discretization (binary)
- Copy proven CD implementation from literature
- Try stronger regularization

### Risk 2: Generated Samples Are Poor Quality
**Expectation**: This is likely! EBMs are hard.
**Response**: 
- Focus on sampling speed, not quality
- Document honestly what works/doesn't
- Still valid to test THRML performance

### Risk 3: THRML Shows No Speedup
**Mitigation**: 
- Try larger problem sizes (upscale MNIST)
- Ensure fair comparison (same algorithm)
- Document overhead sources

### Risk 4: Time Constraints
**Priority Order**:
1. Training working EBM (core requirement)
2. THRML sampling working (core requirement)
3. Speed benchmarks (main goal)
4. Quality metrics (nice to have)
5. Visualizations (polish)

---

## Files to Create

### Core Implementation
- `data_exploration.py` - MNIST analysis
- `preprocessing.py` - Data pipeline
- `ebm_model.py` - EBM architecture
- `thrml_sampler.py` - THRML integration
- `train_ebm.py` - Training loop
- `sample_thrml.py` - THRML sampling
- `sample_naive.py` - Baseline sampling

### Evaluation
- `evaluate_quality.py` - Quality metrics
- `benchmark_sampling.py` - Speed tests
- `convergence_analysis.py` - Convergence study
- `energy_analysis.py` - Energy efficiency

### Visualization
- `visualize_energy.py` - Energy landscape
- `animate_sampling.py` - Sampling dynamics
- `compare_blocking.py` - Block strategies

### Optional
- `baseline_vae.py` - VAE comparison

### Documentation
- `README.md` - Project overview
- `RESULTS.md` - Findings summary
- `PLAN.md` - This file

---

## Expected Timeline

**Optimistic**: 2-3 weeks (focused, no interruptions)  
**Realistic**: 4-6 weeks (with experimentation and debugging)  
**Pessimistic**: 8 weeks (if major issues arise)

---

## Notes & Reminders

- **Keep it simple first**: Don't over-engineer early steps
- **Document as you go**: Write findings when they're fresh
- **Visualize often**: Helps debugging and understanding
- **Honest assessment**: Document failures as much as successes
- **Ask for help**: Use human collaborator for manual tasks
- **Version control**: Commit working states frequently
- **Compare fairly**: Same algorithm for THRML vs naive

---

**Status**: 📋 Planning Complete - Ready to Begin Implementation  
**Next Step**: Phase 1.1 - Data Exploration  
**Updated**: November 4, 2025
