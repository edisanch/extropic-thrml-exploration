#!/usr/bin/env python3
"""
Interactive Ising Model Simulator with Animation
=================================================

Real-time visualization of magnetic spins flipping during Gibbs sampling!
Watch the phase transition happen before your eyes.
"""

import gradio as gr
import jax
import jax.numpy as jnp
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import numpy as np
import io
from PIL import Image


def create_spin_image(spins_2d, title="Spin Configuration"):
    """Create a visual representation of spins as a grid."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Custom colormap: blue for spin down (-1), red for spin up (+1)
    cmap = ListedColormap(['#2E86AB', '#A23B72'])
    
    # Plot the spins
    im = ax.imshow(spins_2d, cmap=cmap, interpolation='nearest', vmin=-1, vmax=1)
    
    # Add grid lines
    for i in range(spins_2d.shape[0] + 1):
        ax.axhline(i - 0.5, color='white', linewidth=1, alpha=0.3)
    for i in range(spins_2d.shape[1] + 1):
        ax.axvline(i - 0.5, color='white', linewidth=1, alpha=0.3)
    
    # Add arrows to show spin direction
    for i in range(spins_2d.shape[0]):
        for j in range(spins_2d.shape[1]):
            if spins_2d[i, j] > 0:
                ax.text(j, i, '↑', ha='center', va='center', 
                       fontsize=20, color='white', weight='bold')
            else:
                ax.text(j, i, '↓', ha='center', va='center', 
                       fontsize=20, color='white', weight='bold')
    
    ax.set_title(title, fontsize=16, weight='bold', pad=20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f'Magnetization: {np.mean(spins_2d):.3f}', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([-1, 1])
    cbar.set_ticklabels(['Spin ↓ (-1)', 'Spin ↑ (+1)'])
    
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img


def animate_ising_sampling(grid_size, coupling, temperature, external_field, n_steps):
    """Run Ising model and return sequence of spin configurations."""
    
    # Create 2D grid of spins
    n_total = grid_size * grid_size
    nodes = [SpinNode() for _ in range(n_total)]
    
    # Create edges for 2D lattice with periodic boundary conditions
    edges = []
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            # Right neighbor (with wrapping)
            right = i * grid_size + ((j + 1) % grid_size)
            edges.append((nodes[idx], nodes[right]))
            # Down neighbor (with wrapping)
            down = ((i + 1) % grid_size) * grid_size + j
            edges.append((nodes[idx], nodes[down]))
    
    biases = jnp.ones((n_total,)) * external_field
    weights = jnp.ones((len(edges),)) * coupling
    beta = 1.0 / max(temperature, 0.01)
    
    model = IsingEBM(nodes, edges, biases, weights, jnp.array(beta))
    
    # Create checkerboard coloring for 2D lattice
    # Color based on (i + j) % 2
    even_indices = [i * grid_size + j for i in range(grid_size) 
                   for j in range(grid_size) if (i + j) % 2 == 0]
    odd_indices = [i * grid_size + j for i in range(grid_size) 
                  for j in range(grid_size) if (i + j) % 2 == 1]
    
    free_blocks = [
        Block([nodes[i] for i in even_indices]),
        Block([nodes[i] for i in odd_indices])
    ]
    
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
    
    # Sample with more steps to see evolution
    key = jax.random.key(42)
    k_init, k_samp = jax.random.split(key, 2)
    init_state = hinton_init(k_init, model, free_blocks, ())
    
    schedule = SamplingSchedule(
        n_warmup=10,
        n_samples=int(n_steps),
        steps_per_sample=1  # Save every step
    )
    
    samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
    spin_samples = samples[0]
    
    # Convert to 2D grid for visualization
    frames = []
    for step_idx in range(min(int(n_steps), spin_samples.shape[0])):
        spins_1d = spin_samples[step_idx]
        spins_2d = spins_1d.reshape(grid_size, grid_size)
        frames.append(np.array(spins_2d))
    
    return frames


def run_animation(grid_size, coupling, temperature, external_field, n_steps):
    """Generate animation frames and return as images."""
    
    frames = animate_ising_sampling(int(grid_size), coupling, temperature, 
                                    external_field, n_steps)
    
    if not frames:
        return None, "No frames generated!"
    
    # Create images for key frames
    images = []
    stats_text = f"## Animation Parameters\n\n"
    stats_text += f"- Grid Size: {int(grid_size)}×{int(grid_size)}\n"
    stats_text += f"- Temperature: {temperature:.2f}\n"
    stats_text += f"- Coupling J: {coupling:.2f}\n"
    stats_text += f"- Total Steps: {len(frames)}\n\n"
    stats_text += "### Frame Selection:\n\n"
    
    # Show initial, middle, and final states
    frame_indices = [0, len(frames)//2, len(frames)-1]
    frame_labels = ["Initial State", "Mid-Evolution", "Final State"]
    
    for idx, label in zip(frame_indices, frame_labels):
        img = create_spin_image(frames[idx], f"{label} (Step {idx})")
        images.append(img)
        mag = np.mean(frames[idx])
        stats_text += f"**{label}**: m = {mag:.3f}\n\n"
    
    # Calculate magnetization evolution
    mags = [np.mean(frame) for frame in frames]
    abs_mags = [abs(m) for m in mags]
    
    stats_text += f"\n### Statistics:\n\n"
    stats_text += f"- Final Magnetization: {mags[-1]:.3f}\n"
    stats_text += f"- Average |m|: {np.mean(abs_mags):.3f}\n"
    stats_text += f"- Std Dev: {np.std(mags):.3f}\n"
    
    if abs(mags[-1]) > 0.7:
        stats_text += "\n**Strong ordering** - Spins aligned! 🧲"
    elif abs(mags[-1]) < 0.3:
        stats_text += "\n**Disordered** - Random spins! 🎲"
    else:
        stats_text += "\n**Partial order** - Near critical point! ⚡"
    
    return images, stats_text


# Create Gradio interface
with gr.Blocks(title="Ising Model with Animation", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧲 Ising Model: Watch Spins Evolve!
    
    **See magnetic spins flip in real-time** during block Gibbs sampling. 
    This is what Extropic's TSU hardware accelerates at the circuit level!
    
    🔴 Red/↑ = Spin Up (+1)  
    🔵 Blue/↓ = Spin Down (-1)
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Parameters")
            
            grid_size = gr.Slider(
                minimum=5, maximum=20, value=10, step=1,
                label="Grid Size (NxN)",
                info="Larger = more spins but slower"
            )
            
            temperature = gr.Slider(
                minimum=0.1, maximum=5.0, value=2.0, step=0.1,
                label="Temperature (T)",
                info="Low T = ordered, High T = random"
            )
            
            coupling = gr.Slider(
                minimum=-2.0, maximum=2.0, value=1.0, step=0.1,
                label="Coupling Strength (J)",
                info="Positive = ferromagnetic"
            )
            
            external_field = gr.Slider(
                minimum=-1.0, maximum=1.0, value=0.0, step=0.1,
                label="External Field (h)"
            )
            
            n_steps = gr.Slider(
                minimum=10, maximum=100, value=50, step=10,
                label="Sampling Steps",
                info="More steps = longer evolution"
            )
            
            run_btn = gr.Button("🎬 Run Animation", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.Markdown("## Evolution Snapshots")
            gallery = gr.Gallery(
                label="Spin Configurations",
                columns=3,
                rows=1,
                height="auto",
                object_fit="contain"
            )
            stats = gr.Markdown("Click 'Run Animation' to see spin evolution!")
    
    gr.Markdown("""
    ---
    ## 🔬 Experiments to Try:
    
    **Phase Transition:**
    - Start at T=3.0 (high) → random spins
    - Lower to T=0.5 (low) → watch spins align!
    
    **Critical Point:**
    - Set T≈2.27 (critical temperature for 2D Ising)
    - See domains form and merge
    
    **Spontaneous Symmetry Breaking:**
    - T=0.5, run multiple times
    - Sometimes all ↑, sometimes all ↓
    
    **Antiferromagnet:**
    - Set J=-1.0 → spins want to anti-align
    - Creates checkerboard pattern!
    """)
    
    run_btn.click(
        fn=run_animation,
        inputs=[grid_size, coupling, temperature, external_field, n_steps],
        outputs=[gallery, stats]
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Ising Model Animation Simulator")
    print("=" * 60)
    print("\nStarting web server with spin visualization...")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True
    )
