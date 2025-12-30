"""
Demo script showing neuron label alignment options.

This script demonstrates:
1. Left-aligned neuron labels
2. Center-aligned neuron labels (default)
3. Right-aligned neuron labels
4. Mixed alignment for different visualization styles
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer, VectorInput
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig

# Create output directory
output_dir = "test_outputs"
os.makedirs(output_dir, exist_ok=True)

print("="*60)
print("Neuron Label Alignment Demo")
print("="*60)

# Example 1: Left-aligned labels
print("\n1. Creating network with LEFT-aligned labels...")
nn_left = NeuralNetwork("Left Alignment")

nn_left.add_layer(VectorInput(
    num_features=4,
    name="Input",
    neuron_labels=["Feature A", "Feature B", "Feature C", "Feature D"],
    label_position="left"
))

nn_left.add_layer(FullyConnectedLayer(5, activation="relu", name="Hidden"))

nn_left.add_layer(FullyConnectedLayer(
    num_neurons=3,
    name="Output",
    neuron_labels=["Class 1", "Class 2", "Class 3"],
    label_position="right"
))

config_left = PlotConfig(
    show_neuron_text_labels=True,
    neuron_text_label_fontsize=11,
    neuron_text_label_alignment='left',  # Left-aligned
    background_color='white'
)

plot_network(
    nn_left,
    config=config_left,
    title="Left-Aligned Labels",
    save_path=os.path.join(output_dir, "demo_alignment_left.png"),
    show=False
)
print("✅ Created: test_outputs/demo_alignment_left.png")

# Example 2: Center-aligned labels (default)
print("\n2. Creating network with CENTER-aligned labels (default)...")
nn_center = NeuralNetwork("Center Alignment")

nn_center.add_layer(VectorInput(
    num_features=4,
    name="Input",
    neuron_labels=["Feature A", "Feature B", "Feature C", "Feature D"],
    label_position="left"
))

nn_center.add_layer(FullyConnectedLayer(5, activation="relu", name="Hidden"))

nn_center.add_layer(FullyConnectedLayer(
    num_neurons=3,
    name="Output",
    neuron_labels=["Class 1", "Class 2", "Class 3"],
    label_position="right"
))

config_center = PlotConfig(
    show_neuron_text_labels=True,
    neuron_text_label_fontsize=11,
    neuron_text_label_alignment='center',  # Center-aligned (default)
    background_color='white'
)

plot_network(
    nn_center,
    config=config_center,
    title="Center-Aligned Labels (Default)",
    save_path=os.path.join(output_dir, "demo_alignment_center.png"),
    show=False
)
print("✅ Created: test_outputs/demo_alignment_center.png")

# Example 3: Right-aligned labels
print("\n3. Creating network with RIGHT-aligned labels...")
nn_right = NeuralNetwork("Right Alignment")

nn_right.add_layer(VectorInput(
    num_features=4,
    name="Input",
    neuron_labels=["Feature A", "Feature B", "Feature C", "Feature D"],
    label_position="left"
))

nn_right.add_layer(FullyConnectedLayer(5, activation="relu", name="Hidden"))

nn_right.add_layer(FullyConnectedLayer(
    num_neurons=3,
    name="Output",
    neuron_labels=["Class 1", "Class 2", "Class 3"],
    label_position="right"
))

config_right = PlotConfig(
    show_neuron_text_labels=True,
    neuron_text_label_fontsize=11,
    neuron_text_label_alignment='right',  # Right-aligned
    background_color='white'
)

plot_network(
    nn_right,
    config=config_right,
    title="Right-Aligned Labels",
    save_path=os.path.join(output_dir, "demo_alignment_right.png"),
    show=False
)
print("✅ Created: test_outputs/demo_alignment_right.png")

# Example 4: LaTeX labels with different alignments
print("\n4. Creating network with LaTeX labels and different alignments...")
nn_latex = NeuralNetwork("LaTeX Alignment")

nn_latex.add_layer(VectorInput(
    num_features=3,
    name="Input",
    neuron_labels=[r"$x_1$", r"$x_2$", r"$x_3$"],
    label_position="left"
))

nn_latex.add_layer(FullyConnectedLayer(4, activation="tanh", name="Hidden"))

nn_latex.add_layer(FullyConnectedLayer(
    num_neurons=2,
    name="Output",
    neuron_labels=[r"$\hat{y}_1$", r"$\hat{y}_2$"],
    label_position="right"
))

# Left-aligned LaTeX
config_latex_left = PlotConfig(
    show_neuron_text_labels=True,
    neuron_text_label_fontsize=12,
    neuron_text_label_alignment='left',
    background_color='white'
)

plot_network(
    nn_latex,
    config=config_latex_left,
    title="LaTeX Labels - Left Aligned",
    save_path=os.path.join(output_dir, "demo_alignment_latex_left.png"),
    show=False
)

# Right-aligned LaTeX
config_latex_right = PlotConfig(
    show_neuron_text_labels=True,
    neuron_text_label_fontsize=12,
    neuron_text_label_alignment='right',
    background_color='white'
)

plot_network(
    nn_latex,
    config=config_latex_right,
    title="LaTeX Labels - Right Aligned",
    save_path=os.path.join(output_dir, "demo_alignment_latex_right.png"),
    show=False
)

print("✅ Created: test_outputs/demo_alignment_latex_left.png")
print("✅ Created: test_outputs/demo_alignment_latex_right.png")

print("\n" + "="*60)
print("Demo complete!")
print("="*60)
print("\nAlignment Options:")
print("  • 'left'   - Labels align on their left edge")
print("  • 'center' - Labels align on their center (default)")
print("  • 'right'  - Labels align on their right edge")
print("\nUsage:")
print("  config = PlotConfig(")
print("      neuron_text_label_alignment='left'  # or 'center' or 'right'")
print("  )")
print("="*60)
