#!/usr/bin/env python3
"""
Interactive Ising Model Simulator - Web Interface
==================================================

Run this to get an interactive web interface where you can:
- Adjust temperature, coupling strength, and other parameters
- See real-time magnetization statistics
- View sample spin configurations
- Explore the phase transition

Usage:
    python ising_interactive.py

Then open your browser to the URL shown (usually http://127.0.0.1:7860)
"""

import gradio as gr
import jax
import jax.numpy as jnp
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init


def run_ising_simulation(n_spins, coupling, temperature, external_field, n_samples):
    """Run Ising model sampling with given parameters."""
    
    # Create the model
    nodes = [SpinNode() for _ in range(n_spins)]
    edges = [(nodes[i], nodes[i+1]) for i in range(n_spins-1)]
    biases = jnp.ones((n_spins,)) * external_field
    weights = jnp.ones((n_spins-1,)) * coupling
    beta = 1.0 / max(temperature, 0.01)  # Avoid division by zero
    
    model = IsingEBM(nodes, edges, biases, weights, jnp.array(beta))
    
    # Create sampling program with 2-coloring
    free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
    
    # Sample
    key = jax.random.key(42)
    k_init, k_samp = jax.random.split(key, 2)
    init_state = hinton_init(k_init, model, free_blocks, ())
    
    schedule = SamplingSchedule(
        n_warmup=100,
        n_samples=int(n_samples),
        steps_per_sample=2
    )
    
    samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
    spin_samples = samples[0]
    
    # Compute statistics
    mean_mag = float(jnp.mean(spin_samples))
    magnetization_per_sample = jnp.mean(spin_samples, axis=1)
    abs_mag = float(jnp.mean(jnp.abs(magnetization_per_sample)))
    std_mag = float(jnp.std(magnetization_per_sample))
    
    # Format output
    stats_text = f"""
## Statistics

- **Mean Magnetization**: {mean_mag:.4f}
- **|Magnetization|**: {abs_mag:.4f}  (0 = random, 1 = fully aligned)
- **Std Deviation**: {std_mag:.4f}
- **Temperature**: {temperature:.3f}
- **Coupling J**: {coupling:.3f}
- **External Field h**: {external_field:.3f}

### Physical Interpretation:
"""
    
    if abs_mag > 0.7:
        stats_text += "**Strong magnetization** - spins are highly aligned! (Low T or strong coupling)"
    elif abs_mag > 0.3:
        stats_text += "**Partial magnetization** - some alignment present (Near critical point)"
    else:
        stats_text += "**Weak magnetization** - spins are mostly random (High T)"
    
    # Show sample configurations
    samples_text = "\n### Sample Spin Configurations:\n\n"
    for i in range(min(10, int(n_samples))):
        spins_str = ''.join(['↑' if s > 0 else '↓' for s in spin_samples[i]])
        mag = magnetization_per_sample[i]
        samples_text += f"`{spins_str}`  (m={mag:.2f})\n\n"
    
    return stats_text + samples_text


# Create Gradio interface
with gr.Blocks(title="Ising Model Explorer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧲 Interactive Ising Model Simulator
    
    Explore magnetic phase transitions using THRML's block Gibbs sampling!
    
    **What is this?** The Ising model describes spins (tiny magnets) on a chain. 
    At high temperatures, spins are random. At low temperatures, they align spontaneously.
    This is what Extropic's TSU hardware accelerates!
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Parameters")
            
            n_spins = gr.Slider(
                minimum=5, maximum=50, value=20, step=1,
                label="Number of Spins",
                info="Chain length (5-50)"
            )
            
            temperature = gr.Slider(
                minimum=0.1, maximum=5.0, value=1.0, step=0.1,
                label="Temperature (T)",
                info="High T = random, Low T = aligned"
            )
            
            coupling = gr.Slider(
                minimum=-2.0, maximum=2.0, value=0.5, step=0.1,
                label="Coupling Strength (J)",
                info="Positive = ferromagnetic (align), Negative = antiferromagnetic"
            )
            
            external_field = gr.Slider(
                minimum=-1.0, maximum=1.0, value=0.0, step=0.1,
                label="External Field (h)",
                info="Bias toward ↑ (positive) or ↓ (negative)"
            )
            
            n_samples = gr.Slider(
                minimum=100, maximum=2000, value=500, step=100,
                label="Number of Samples",
                info="More = better statistics (but slower)"
            )
            
            run_btn = gr.Button("🚀 Run Simulation", variant="primary", size="lg")
        
        with gr.Column():
            gr.Markdown("## Results")
            output = gr.Markdown("Click 'Run Simulation' to start!")
    
    gr.Markdown("""
    ---
    ### Tips for Exploration:
    
    **To see phase transition:**
    1. Set J=1.0, h=0.0, N=30
    2. Start with T=3.0 (random) → observe low |magnetization|
    3. Decrease T to 0.5 (ordered) → observe high |magnetization|
    4. Watch the transition happen around T ≈ 1.0!
    
    **To see spontaneous symmetry breaking:**
    - Run multiple times with same low T → sometimes ↑↑↑, sometimes ↓↓↓
    
    **To see frustration (antiferromagnet):**
    - Set J=-1.0, T=1.0 → spins want to anti-align: ↑↓↑↓↑↓
    """)
    
    run_btn.click(
        fn=run_ising_simulation,
        inputs=[n_spins, coupling, temperature, external_field, n_samples],
        outputs=output
    )

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Interactive Ising Model Simulator")
    print("=" * 60)
    print("\nStarting web server...")
    
    # Launch with explicit settings
    demo.launch(
        server_name="0.0.0.0",  # Allow connections from anywhere
        server_port=7860,
        share=False,
        inbrowser=True,  # Try to open browser automatically
        show_error=True
    )
