# Copilot Instructions for Extropic THRML Exploration

## Project Overview
Hands-on validation workspace for Extropic's thermodynamic computing claims through probabilistic graphical models. **Critical approach**: Experimental validation with healthy skepticism, not blind advocacy. Document both successes and failures.

## Communication Guidelines

**Do NOT create separate markdown files for step summaries or documentation** unless explicitly requested. Instead:
- Provide summaries directly in the chat response
- Keep responses concise and focused
- Update existing documentation files (PLAN.md, README.md) as needed
- Only create new files for actual implementation (code, tests, benchmarks)

## Environment Setup

**CRITICAL**: Always activate the virtual environment before running any Python commands:
```bash
source .venv/bin/activate
```

This must be done at the start of every terminal session. The venv contains all required dependencies including JAX, THRML, PyTorch, and TensorBoard.

## Core Technology Stack
- **THRML** (Thermodynamic Hypergraphical Model Library): JAX-based block Gibbs sampling
- **JAX/JAXlib**: GPU-accelerated probabilistic computing with JIT compilation
- **PyTorch**: EBM training (gradient computation only; sampling uses THRML)
- **Key principle**: PyTorch for gradients, JAX/THRML for sampling - never mix in same computation

## Project Structure

### 01_ising_phase_transitions/ ✅ Complete
Simple Ising model explorer for understanding block Gibbs fundamentals. Key learnings:
- **Crossover point**: ~500 spins where THRML beats naive Python (between 16×16 and 32×32 grids)
- Below 500 spins: Naive Python wins due to JAX/GPU overhead
- GPU surprisingly slower than CPU at these scales (memory transfer overhead dominates)
- Scripts follow pattern: `ising_basic.py` (1D) → `ising_animated.py` (2D viz) → `phase_transition.py` (temp sweep) → `benchmark.py` (performance)

### 02_ebm_mnist_generation/ 🔄 Active Development
Training Energy-Based Models on MNIST to validate generative model efficiency claims.

**Current Status** (see PLAN.md):
- ✅ Phase 1-3: Data prep, EBM architecture, training with TensorBoard monitoring
- 🚀 Phase 4: Critical optimization path (JIT + GPU + vectorization for <1ms/sample target)
- ⏳ Phase 5-9: Hyperparameter tuning, multi-digit training, quality benchmarking

**Architecture decisions**:
- Discretization: 4 levels `{0,1,2,3}` (balance of detail vs complexity)
- Energy function: Potts model with 3,138 params (3,136 biases + 2 edge weights)
- Block structure: 2-color checkerboard (392 pixels/block) for 4-neighbor grid
- Cached data: 418.7 MB preprocessed arrays in `preprocessed_cache/` (gitignored)

## Development Patterns

### THRML Sampling Architecture
Block Gibbs requires proper factor graph setup. See `thrml_sampler_native.py` for canonical pattern:

```python
# ✅ CORRECT: Native THRML components (optimized)
from thrml.models import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml.factor import FactorSamplingProgram

# Create factors (vectorized energy)
bias_factor = CategoricalEBMFactor(
    node_groups=[Block(nodes)],
    weights=bias_params  # (784, 4) - all nodes, all levels
)

# Create edge factors with Potts matrices
h_edge_factor = CategoricalEBMFactor(
    node_groups=[Block(h_left_nodes), Block(h_right_nodes)],
    weights=h_edge_weights  # (756, 4, 4) - Potts interaction matrices
)

# Build sampling program
program = FactorSamplingProgram(
    gibbs_spec=spec,
    samplers=[CategoricalGibbsConditional(n_levels=4) for _ in blocks],
    factors=[bias_factor, h_edge_factor, v_edge_factor],
    other_interaction_groups=[]
)

# JIT-compiled sampling (first call compiles, subsequent calls <1ms)
samples = sample_states(key, program, schedule, init_state, [], [Block(nodes)])
```

**Common pitfalls**:
- ❌ Don't use Python loops for Gibbs updates - let THRML handle iteration
- ❌ Don't mix PyTorch and JAX in same operation (convert parameters explicitly)
- ❌ Don't forget JIT warmup - first call includes compilation overhead (~500ms)

### Performance Optimization Priority
When optimizing sampling (see PLAN.md Phase 4):
1. **Native THRML components** - use `CategoricalEBMFactor` not custom loops
2. **Vectorization** - `jax.vmap` for batch parallelism
3. **JIT compilation** - `@jax.jit` for compilation
4. **GPU placement** - verify with `jax.devices()`, avoid CPU↔GPU transfers

**When in doubt**: Consult official THRML documentation at https://docs.thrml.ai and https://github.com/extropic-ai/thrml

### Energy Function Conventions
All EBMs follow **negative energy** convention for consistency:
```python
# Potts model energy
E(x; θ) = -Σ_i bias_i[x_i] - Σ_{<i,j>} weight_type * δ(x_i, x_j)

# Where δ(x_i, x_j) = 1 if same level (ferromagnetic coupling)
# Lower energy = more probable state
```

PyTorch EBM stores biases as `(H, W, n_levels)`, THRML expects `(n_pixels, n_levels)` - always flatten and convert to JAX via `convert_ebm_to_thrml_parameters()`.

### Training Patterns
Contrastive Divergence (CD-1) workflow:
```python
# Positive phase: real data
energies_data = ebm(real_images)

# Negative phase: samples from model (requires fast sampling!)
# Initially used random negatives; need optimized THRML for true Gibbs negatives
samples = thrml_sampler.sample_batch(batch_size, n_steps=config.cd_steps)
energies_samples = ebm(samples)

# CD gradient: ∇_θ E_data - ∇_θ E_samples
loss = energies_data.mean() - energies_samples.mean()
```

Training currently uses `use_gibbs_in_training=False` (random negatives) due to sampling speed bottleneck. Phase 4 optimization will enable proper Gibbs sampling.

### File Naming Conventions
- `*_basic.py`: Minimal working examples (start here)
- `*_interactive.py`: Gradio/Matplotlib interactive GUIs
- `*_animated.py`: Real-time visualization with animation
- `benchmark*.py`: Performance measurement scripts
- `test_*.py`: Unit tests and validation
- `PLAN.md`: Detailed implementation roadmap with status tracking

## Critical Developer Knowledge

### Workspace Setup
```bash
# Always use venv (never system Python)
source .venv/bin/activate

# MNIST cache creation (one-time, creates 418.7 MB)
cd 02_ebm_mnist_generation
python create_cache.py  # Required before training

# Training with monitoring
python train_ebm_monitored.py  # TensorBoard logs to runs/

# View training progress
tensorboard --logdir=runs/
```

### Debugging Performance Issues
1. Check JAX backend: `import jax; print(jax.devices())` - should show GPU if available
2. Profile JIT compilation: First run includes compilation time
3. Benchmark sampling: Compare baseline vs optimized implementations
4. GPU utilization: `nvidia-smi -l 1` in separate terminal during training

**Reference**: Official THRML documentation at https://docs.thrml.ai for API details and best practices.

### Common Issues
- **"No module named 'thrml'"**: Forgot to activate `.venv`
- **Energy mismatch between PyTorch/JAX**: Check parameter conversion in `convert_ebm_to_thrml_parameters()`
- **Slow sampling (>20ms/step)**: Using custom sampler instead of native THRML - migrate to `thrml_sampler_native.py`
- **JIT recompilation on every call**: Batch size or shape changed - consistent shapes = one compilation
- **Out of memory**: Reduce batch size or use CPU backend temporarily

## Testing Conventions
Each phase has validation script:
- `data_exploration.py`: Visual data inspection before processing
- `test_gradio.py`: Verify interactive components work
- `benchmark.py`: Performance comparison against baselines
- `test_trained_model.py`: Validate loaded checkpoints

Follow standard scientific practice: verify correctness before optimizing, document methodology, and measure results reproducibly.

## Architecture Decision Log

### Why 4-level discretization?
Tested 2, 4, 16 levels. Binary loses detail; 16-level increases sampling complexity 4x. 4-level is sweet spot (see `data_exploration.py` analysis).

### Why 2-coloring instead of 4-coloring?
For 4-neighbor grids, 2-coloring is optimal (392 pixels/block). 4-coloring provides no parallelism benefit, only comparison value (see `block_comparison.py`).

### Why separate horizontal/vertical edge weights?
Allows learning anisotropic correlations (e.g., MNIST digits have vertical elongation). Adds only 1 extra parameter vs uniform weights.

### Why PyTorch for training, JAX for sampling?
PyTorch has mature autograd for EBM gradients; JAX excels at parallel sampling. Playing to each framework's strengths. Parameter conversion overhead is negligible (~1ms).

## Knowledge Base Reference
See `Extropic_Knowledge_Base.md` for comprehensive Extropic technology context, including critical analysis section (⚠️) covering realistic scenarios and limitations. Always reference when discussing efficiency claims.

## Contribution Guidelines
- **Document honestly**: Mark both successes (✅) and blockers (🚀) in PLAN.md
- **Measure everything**: Before/after benchmarks for optimizations, save plots
- **Update status**: Keep PLAN.md current with phase completion and findings
- **Commit working states**: Checkpoint after each phase completion
- **Reality-check claims**: Compare against published 10,000x efficiency with full context
