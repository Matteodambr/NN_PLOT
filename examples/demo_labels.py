"""
Quick demo showing custom neuron labels with LaTeX support.
Creates a simple but comprehensive example.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer
from NN_PLOTTING_UTILITIES import plot_network, PlotConfig, LayerStyle

os.makedirs("test_outputs", exist_ok=True)

# Example 1: Realistic machine learning model with descriptive labels
print("Creating labeled neural network example...")

nn = NeuralNetwork("Customer Churn Prediction")

# Input layer - features with plain text labels on the left
nn.add_layer(FullyConnectedLayer(
    num_neurons=5,
    name="Input Features",
    neuron_labels=[
        "Monthly Charges",
        "Contract Length",
        "Support Tickets",
        "Usage Minutes",
        "Customer Age"
    ],
    label_position="left"
))

# Hidden layers - no labels needed
nn.add_layer(FullyConnectedLayer(
    num_neurons=8,
    activation="relu",
    name="Hidden Layer 1"
))

nn.add_layer(FullyConnectedLayer(
    num_neurons=4,
    activation="relu",
    name="Hidden Layer 2"
))

# Output layer - predictions with labels on the right
nn.add_layer(FullyConnectedLayer(
    num_neurons=2,
    activation="softmax",
    name="Churn Prediction",
    neuron_labels=["Will Stay", "Will Churn"],
    label_position="right"
))

# Create plot with labels enabled
config = PlotConfig(
    show_neuron_text_labels=True,
    neuron_text_label_fontsize=10,
    figsize=(14, 8)
)

plot_network(
    nn,
    config=config,
    title="Customer Churn Prediction Model",
    save_path="test_outputs/demo_labeled_network.png",
    show=False,
    dpi=300
)

print("✅ Created: test_outputs/demo_labeled_network.png")

# Example 2: LaTeX mathematical notation
print("Creating LaTeX math example...")

nn_math = NeuralNetwork("Mathematical Model")

# Input with LaTeX math notation
nn_math.add_layer(FullyConnectedLayer(
    num_neurons=3,
    name="Input",
    neuron_labels=[r"$x_1$", r"$x_2$", r"$x_3$"],
    label_position="left"
))

nn_math.add_layer(FullyConnectedLayer(
    num_neurons=5,
    activation="tanh",
    name="Hidden",
    neuron_labels=[r"$h_1$", r"$h_2$", r"$h_3$", r"$h_4$", r"$h_5$"],
    label_position="left"
))

# Output with predicted values
nn_math.add_layer(FullyConnectedLayer(
    num_neurons=2,
    activation="sigmoid",
    name="Output",
    neuron_labels=[r"$\hat{y}_1$", r"$\hat{y}_2$"],
    label_position="right"
))

config_math = PlotConfig(
    show_neuron_text_labels=True,
    neuron_text_label_fontsize=12
)

plot_network(
    nn_math,
    config=config_math,
    title="Neural Network with LaTeX Notation",
    save_path="test_outputs/demo_latex_labels.png",
    show=False,
    dpi=300
)

# Also save as SVG for presentations
plot_network(
    nn_math,
    config=config_math,
    title="Neural Network with LaTeX Notation",
    save_path="test_outputs/demo_latex_labels.svg",
    show=False,
    format="svg"
)

print("✅ Created: test_outputs/demo_latex_labels.png")
print("✅ Created: test_outputs/demo_latex_labels.svg (scalable vector)")

# Example 3: All three label types together
print("\nCreating comprehensive labeling example (all three types)...")

nn_all = NeuralNetwork("Complete Labeling Demo")

# Input layer with neuron labels
input_layer = FullyConnectedLayer(
    num_neurons=4,
    name="Input Layer",
    neuron_labels=["Age", "Income", "Credit Score", "Debt Ratio"],
    label_position="left"
)
nn_all.add_layer(input_layer)

# Hidden layers
nn_all.add_layer(FullyConnectedLayer(
    num_neurons=6,
    activation="relu",
    name="Hidden Layer 1"
))

nn_all.add_layer(FullyConnectedLayer(
    num_neurons=4,
    activation="relu",
    name="Hidden Layer 2"
))

# Output layer with neuron labels
output_layer = FullyConnectedLayer(
    num_neurons=2,
    activation="softmax",
    name="Output Layer",
    neuron_labels=["Approved", "Denied"],
    label_position="right"
)
nn_all.add_layer(output_layer)

# Configure with ALL three label types enabled
config_all = PlotConfig(
    figsize=(16, 8),
    background_color='white',
    # 1. Neuron labels (individual neuron text)
    show_neuron_text_labels=True,
    neuron_text_label_fontsize=20,
    
    # 2. Layer names (layer titles below each layer)
    show_layer_names=True,
    layer_name_fontsize=11,
    layer_names_custom={  # Custom layer label text (Line 1)
        'Input Layer': 'Input',
        'Hidden Layer 2': 'Processor',
        'Output Layer': 'Output'
    },
    layer_names_show_type=True,  # Show layer type like "Dense" (Line 2)
    layer_names_show_dim=True,  # Show dimension info (e.g., "Dim.: 4") (Line 3)
    layer_names_show_activation=True,  # Show activation (e.g., "Act.: ReLU") (Line 4)
    layer_names_align_bottom=True,  # All layer labels at same height
    layer_names_bottom_offset=1,  # Distance from bottom
    layer_names_show_box=False,  # No boxes around labels
    layer_names_line_styles=['curly_brace'],  # Curly brace pointing down
    layer_names_line_color='black',  # Color of the brace
    layer_names_line_width=2,  # Thickness of the brace
    layer_names_brace_width_multiplier=1.5,  # Make braces 50% wider
    
    # Title positioning
    title_offset=10,  # Distance from top
    
    # 3. Layer variable names (high-level input/output descriptions)
    show_layer_variable_names=True,
    layer_variable_names={
        'Input Layer': 'Customer Financial Profile Information',
        'Output Layer': 'Loan Decision Classification'
    },
    layer_variable_names_fontsize=12,
    layer_variable_names_position='side',
    layer_variable_names_wrap=True,  # Enable text wrapping
    layer_variable_names_max_width=15,  # Wrap at 15 characters
    layer_variable_names_multialignment='center',  # Center-align wrapped text
    layer_variable_names_offset=1.0,  # Distance from layer edges
    
    # Optional: Add styling for clarity
    layer_styles={
        'Input Layer': LayerStyle(
            neuron_fill_color='lightblue',
            box_around_layer=True,
            box_fill_color='#E0F2FF',
            box_edge_color='darkblue',
            box_edge_width=2.0,
            show_activation=False  # Override: don't show activation for input layer
        ),
        'Hidden Layer 1': LayerStyle(
            show_type=False,  # Override: hide "FC layer" for this specific layer
            show_dim=True,    # Keep dimension visible
            show_activation=True  # Keep activation visible
        ),
        'Output Layer': LayerStyle(
            neuron_fill_color='lightcoral',
            box_around_layer=True,
            box_fill_color='#FFE4E1',
            box_edge_color='darkred',
            box_edge_width=2.0
        )
    }
)

plot_network(
    nn_all,
    config=config_all,
    title="All Three Label Types Demonstrated",
    save_path="test_outputs/demo_all_label_types.png",
    show=False,
    dpi=300
)

print("✅ Created: test_outputs/demo_all_label_types.png")

print("\n" + "="*60)
print("Demo complete! Check the 'test_outputs' folder for:")
print("  1. demo_labeled_network.png - Real-world example with text labels")
print("  2. demo_latex_labels.png - LaTeX mathematical notation")
print("  3. demo_latex_labels.svg - Scalable vector version")
print("  4. demo_all_label_types.png - ALL THREE LABEL TYPES TOGETHER:")
print("     • Neuron labels (individual neuron text)")
print("     • Layer names (layer titles)")
print("     • Layer variable names (input/output descriptions)")
print("="*60)
