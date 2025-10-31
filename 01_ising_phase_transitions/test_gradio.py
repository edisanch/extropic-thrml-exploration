#!/usr/bin/env python3
"""
Test if Gradio is working properly
"""
import gradio as gr

def greet(name):
    return f"Hello {name}!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")

if __name__ == "__main__":
    print("Testing Gradio...")
    demo.launch(server_name="127.0.0.1", server_port=7860)
