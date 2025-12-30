"""
Test script to check if bold math text is supported in neuron labels.
Tests various bold LaTeX commands.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer, VectorInput
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig

# Create output directory at project root
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
output_dir = os.path.join(project_root, "test_outputs")
os.makedirs(output_dir, exist_ok=True)

print("Testing bold math text support...")

nn = NeuralNetwork("Bold Math Test")

# Try different bold math commands
nn.add_layer(VectorInput(
    num_features=6,
    name="Bold Math Tests",
    neuron_labels=[
        r"$\mathbf{x}_1$",      # \mathbf (bold)
        r"$\boldsymbol{x}_2$",  # \boldsymbol (bold + italic)
        r"$\mathbf{X}$",        # Bold capital letter
        r"$\boldsymbol{\alpha}$",  # Bold Greek letter
        r"$\mathbf{h}_i$",      # Bold with subscript
        r"$\boldsymbol{\theta}$"   # Bold theta
    ],
    label_position="left"
))

nn.add_layer(FullyConnectedLayer(
    num_neurons=3,
    activation="relu",
    name="Hidden"
))

nn.add_layer(FullyConnectedLayer(
    num_neurons=2,
    name="Output",
    neuron_labels=[
        r"$\mathbf{y}_1$",
        r"$\boldsymbol{\hat{y}}_2$"
    ],
    label_position="right"
))

config = PlotConfig(
    show_neuron_text_labels=True,
    neuron_text_label_fontsize=12
)

try:
    plot_network(
        nn,
        config=config,
        title="Bold Math Text Test",
        save_path=os.path.join(output_dir, r"test_bold_math.png"),
        show=False,
        dpi=300
    )
    print("✅ Bold math text is SUPPORTED!")
    print("   Created: test_outputs/test_bold_math.png")
    print("\nSupported commands:")
    print("  - \\mathbf{x}: Bold letter")
    print("  - \\boldsymbol{x}: Bold + italic")
    print("  - \\boldsymbol{\\alpha}: Bold Greek letters")
    
except Exception as e:
    print(f"❌ Bold math text may have issues: {e}")

# Test comparison: regular vs bold
print("\nCreating comparison plot...")
nn_compare = NeuralNetwork("Regular vs Bold")

nn_compare.add_layer(VectorInput(
    num_features=4,
    name="Input",
    neuron_labels=[
        r"$x_1$ (regular)",
        r"$\mathbf{x}_2$ (bold)",
        r"$\alpha$ (regular)",
        r"$\boldsymbol{\alpha}$ (bold)"
    ],
    label_position="left"
))

nn_compare.add_layer(FullyConnectedLayer(3, activation="relu", name="Hidden"))

nn_compare.add_layer(FullyConnectedLayer(
    num_neurons=2,
    name="Output",
    neuron_labels=[r"$y$", r"$\mathbf{y}$"],
    label_position="right"
))

plot_network(
    nn_compare,
    config=config,
    title="Regular vs Bold Math Comparison",
    save_path=os.path.join(output_dir, r"test_regular_vs_bold.png"),
    show=False,
    dpi=300
)

print("✅ Created: test_outputs/test_regular_vs_bold.png")
print("\nYou can visually compare regular and bold math text!")
