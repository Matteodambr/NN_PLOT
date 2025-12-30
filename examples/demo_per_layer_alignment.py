"""
Demo script showing per-layer neuron label alignment control.

This script demonstrates:
1. Different alignment for different layers in the same network
2. Overriding global alignment with layer-specific settings
3. Practical use cases for mixed alignments
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer, VectorInput
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig, LayerStyle

# Create output directory
output_dir = "test_outputs"
os.makedirs(output_dir, exist_ok=True)

print("="*60)
print("Per-Layer Label Alignment Demo")
print("="*60)

# Example 1: Different alignments for input vs output layers
print("\n1. Creating network with different alignments per layer...")
nn = NeuralNetwork("Per-Layer Alignment")

nn.add_layer(VectorInput(
    num_features=4,
    name="Input",
    neuron_labels=["Age", "Income", "Credit Score", "Debt Ratio"],
    label_position="left"
))

nn.add_layer(FullyConnectedLayer(5, activation="relu", name="Hidden"))

nn.add_layer(FullyConnectedLayer(
    num_neurons=3,
    name="Output",
    neuron_labels=["Low Risk", "Medium Risk", "High Risk"],
    label_position="right"
))

# Global alignment is 'center', but override for specific layers
config = PlotConfig(
    show_neuron_text_labels=True,
    neuron_text_label_fontsize=11,
    neuron_text_label_alignment='center',  # Global default
    background_color='white',
    layer_styles={
        'Input': LayerStyle(
            neuron_fill_color='lightblue',
            neuron_text_label_alignment='left'  # Input labels left-aligned
        ),
        'Output': LayerStyle(
            neuron_fill_color='lightcoral',
            neuron_text_label_alignment='right'  # Output labels right-aligned
        )
    }
)

plot_network(
    nn,
    config=config,
    title="Per-Layer Alignment: Input=Left, Output=Right",
    save_path=os.path.join(output_dir, "demo_per_layer_alignment_mixed.png"),
    show=False
)
print("✅ Created: test_outputs/demo_per_layer_alignment_mixed.png")

# Example 2: LaTeX labels with per-layer alignment
print("\n2. Creating network with LaTeX and per-layer alignments...")
nn_latex = NeuralNetwork("LaTeX Per-Layer")

nn_latex.add_layer(VectorInput(
    num_features=3,
    name="Input",
    neuron_labels=[r"$x_1$", r"$x_2$", r"$x_3$"],
    label_position="left"
))

nn_latex.add_layer(FullyConnectedLayer(
    num_neurons=5,
    activation="tanh",
    name="Hidden",
    neuron_labels=[r"$h_1$", r"$h_2$", r"$h_3$", r"$h_4$", r"$h_5$"],
    label_position="left"
))

nn_latex.add_layer(FullyConnectedLayer(
    num_neurons=2,
    name="Output",
    neuron_labels=[r"$\hat{y}_1$", r"$\hat{y}_2$"],
    label_position="right"
))

config_latex = PlotConfig(
    show_neuron_text_labels=True,
    neuron_text_label_fontsize=12,
    neuron_text_label_alignment='center',  # Global default
    background_color='white',
    layer_styles={
        'Input': LayerStyle(
            neuron_fill_color='#E3F2FD',
            neuron_edge_color='#1976D2',
            neuron_text_label_alignment='right'  # Right-align input labels
        ),
        'Hidden': LayerStyle(
            neuron_fill_color='#F3E5F5',
            neuron_edge_color='#7B1FA2',
            neuron_text_label_alignment='left'  # Left-align hidden labels
        ),
        'Output': LayerStyle(
            neuron_fill_color='#FFEBEE',
            neuron_edge_color='#C62828',
            neuron_text_label_alignment='center'  # Center-align output labels
        )
    }
)

plot_network(
    nn_latex,
    config=config_latex,
    title="LaTeX Labels with Different Alignments Per Layer",
    save_path=os.path.join(output_dir, "demo_per_layer_latex_mixed.png"),
    show=False
)
print("✅ Created: test_outputs/demo_per_layer_latex_mixed.png")

# Example 3: Practical use case - emphasizing input features
print("\n3. Creating practical example with emphasized input features...")
nn_practical = NeuralNetwork("Credit Risk Model")

nn_practical.add_layer(VectorInput(
    num_features=5,
    name="Features",
    neuron_labels=[
        "Annual Income",
        "Credit History",
        "Loan Amount",
        "Employment Status",
        "Debt-to-Income"
    ],
    label_position="left"
))

nn_practical.add_layer(FullyConnectedLayer(6, activation="relu", name="Processing"))

nn_practical.add_layer(FullyConnectedLayer(
    num_neurons=2,
    name="Decision",
    neuron_labels=["Approve", "Deny"],
    label_position="right"
))

config_practical = PlotConfig(
    show_neuron_text_labels=True,
    neuron_text_label_fontsize=10,
    neuron_text_label_alignment='center',
    background_color='white',
    figsize=(14, 8),
    layer_styles={
        'Features': LayerStyle(
            neuron_fill_color='lightblue',
            box_around_layer=True,
            box_fill_color='#E0F2FF',
            box_edge_color='darkblue',
            box_edge_width=2.0,
            neuron_text_label_alignment='right'  # Right-align to emphasize feature names
        ),
        'Decision': LayerStyle(
            neuron_fill_color='lightcoral',
            box_around_layer=True,
            box_fill_color='#FFE4E1',
            box_edge_color='darkred',
            box_edge_width=2.0,
            neuron_text_label_alignment='left'  # Left-align for clean output labels
        )
    }
)

plot_network(
    nn_practical,
    config=config_practical,
    title="Credit Risk Model - Per-Layer Label Alignment",
    save_path=os.path.join(output_dir, "demo_per_layer_practical.png"),
    show=False,
    dpi=300
)
print("✅ Created: test_outputs/demo_per_layer_practical.png")

print("\n" + "="*60)
print("Demo complete!")
print("="*60)
print("\nPer-Layer Alignment Control:")
print("  • Set global default: config.neuron_text_label_alignment")
print("  • Override per layer: LayerStyle(neuron_text_label_alignment='...')")
print("  • Options: 'left', 'center', 'right'")
print("\nExample Usage:")
print("  config = PlotConfig(")
print("      neuron_text_label_alignment='center',  # Global default")
print("      layer_styles={")
print("          'Input': LayerStyle(")
print("              neuron_text_label_alignment='left'  # Override for Input")
print("          ),")
print("          'Output': LayerStyle(")
print("              neuron_text_label_alignment='right'  # Override for Output")
print("          )")
print("      }")
print("  )")
print("="*60)
