"""
Demonstration of layer variable names feature.

This example shows how to add variable name labels to input and output layers,
which is useful for documenting what each layer represents in your network.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig, LayerStyle

# Create output directory
output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("Variable Names Feature Demonstration")
print("=" * 70)

# ==============================================================================
# Example 1: Simple network with variable names on the side (default)
# ==============================================================================
print("\n[Example 1] Simple network with side-positioned variable names")
print("-" * 70)

nn1 = NeuralNetwork("Simple Classifier")

input_layer = FullyConnectedLayer(num_neurons=4, name="Input")
nn1.add_layer(input_layer)

hidden = FullyConnectedLayer(num_neurons=8, activation="relu", name="Hidden")
nn1.add_layer(hidden, parent_ids=[input_layer.layer_id])

output = FullyConnectedLayer(num_neurons=3, activation="softmax", name="Output")
nn1.add_layer(output, parent_ids=[hidden.layer_id])

config1 = PlotConfig(
    figsize=(12, 6),
    show_title=True,
    show_layer_names=True,
    layer_variable_names={
        'Input': r'$x_1, x_2, x_3, x_4$ (Features)',
        'Output': r'Class A, B, C'
    },
    show_layer_variable_names=True,
    layer_variable_names_fontsize=10,
    layer_variable_names_position='side'  # Default: left for input, right for output
)

plot_network(nn1, config=config1, 
             save_path=os.path.join(output_dir, "variable_names_side.png"),
             show=False)
print("✓ Saved: variable_names_side.png")

# ==============================================================================
# Example 2: Variable names positioned above layers
# ==============================================================================
print("\n[Example 2] Variable names positioned above layers")
print("-" * 70)

nn2 = NeuralNetwork("Image Classifier")

input_layer2 = FullyConnectedLayer(num_neurons=784, name="Input")
nn2.add_layer(input_layer2)

hidden2 = FullyConnectedLayer(num_neurons=128, activation="relu", name="Hidden")
nn2.add_layer(hidden2, parent_ids=[input_layer2.layer_id])

output2 = FullyConnectedLayer(num_neurons=10, activation="softmax", name="Output")
nn2.add_layer(output2, parent_ids=[hidden2.layer_id])

config2 = PlotConfig(
    figsize=(12, 6),
    show_title=True,
    show_layer_names=True,
    max_neurons_per_layer=6,
    layer_variable_names={
        'Input': '28×28 Image Pixels',
        'Output': 'Digits: 0-9'
    },
    show_layer_variable_names=True,
    layer_variable_names_fontsize=11,
    layer_variable_names_position='above'
)

plot_network(nn2, config=config2,
             save_path=os.path.join(output_dir, "variable_names_above.png"),
             show=False)
print("✓ Saved: variable_names_above.png")

# ==============================================================================
# Example 3: Variable names positioned below layers
# ==============================================================================
print("\n[Example 3] Variable names positioned below layers")
print("-" * 70)

config3 = PlotConfig(
    figsize=(12, 6),
    show_title=True,
    show_layer_names=True,
    max_neurons_per_layer=6,
    layer_variable_names={
        'Input': 'Raw Pixel Values',
        'Output': 'Predicted Classes'
    },
    show_layer_variable_names=True,
    layer_variable_names_fontsize=11,
    layer_variable_names_position='below'
)

plot_network(nn2, config=config3,
             save_path=os.path.join(output_dir, "variable_names_below.png"),
             show=False)
print("✓ Saved: variable_names_below.png")

# ==============================================================================
# Example 4: LaTeX-formatted variable names with mathematical notation
# ==============================================================================
print("\n[Example 4] LaTeX-formatted mathematical variable names")
print("-" * 70)

nn4 = NeuralNetwork("Regression Model")

input_layer4 = FullyConnectedLayer(num_neurons=3, name="Input")
nn4.add_layer(input_layer4)

hidden4 = FullyConnectedLayer(num_neurons=10, activation="tanh", name="Hidden")
nn4.add_layer(hidden4, parent_ids=[input_layer4.layer_id])

output4 = FullyConnectedLayer(num_neurons=1, name="Output")
nn4.add_layer(output4, parent_ids=[hidden4.layer_id])

config4 = PlotConfig(
    figsize=(12, 6),
    show_title=True,
    show_layer_names=True,
    layer_variable_names={
        'Input': r'$\mathbf{x} = [x_1, x_2, x_3]^T$',
        'Output': r'$\hat{y} \in \mathbb{R}$'
    },
    show_layer_variable_names=True,
    layer_variable_names_fontsize=12,
    layer_variable_names_position='side'
)

plot_network(nn4, config=config4,
             save_path=os.path.join(output_dir, "variable_names_latex.png"),
             show=False)
print("✓ Saved: variable_names_latex.png")

# ==============================================================================
# Example 5: Multi-output network with variable names (like policy network)
# ==============================================================================
print("\n[Example 5] Multi-output network with variable names")
print("-" * 70)

nn5 = NeuralNetwork("Multi-Task Network")

input_layer5 = FullyConnectedLayer(num_neurons=6, name="Input")
nn5.add_layer(input_layer5)

hidden5 = FullyConnectedLayer(num_neurons=100, activation="relu", name="Hidden")
nn5.add_layer(hidden5, parent_ids=[input_layer5.layer_id])

output_head1 = FullyConnectedLayer(num_neurons=5, activation="softmax", name="Output_Head_1")
nn5.add_layer(output_head1, parent_ids=[hidden5.layer_id])

output_head2 = FullyConnectedLayer(num_neurons=5, activation="softmax", name="Output_Head_2")
nn5.add_layer(output_head2, parent_ids=[hidden5.layer_id])

config5 = PlotConfig(
    figsize=(14, 8),
    show_title=True,
    show_layer_names=True,
    max_neurons_per_layer=6,
    layer_variable_names={
        'Input': 'State: position, velocity, angle',
        'Output_Head_1': 'Action A: move left/right',
        'Output_Head_2': 'Action B: jump/duck'
    },
    show_layer_variable_names=True,
    layer_variable_names_fontsize=10,
    layer_variable_names_position='side',
    layer_styles={
        'Input': LayerStyle(
            neuron_fill_color='#FFD700',
            box_around_layer=True,
            box_fill_color='#FFFACD',
            box_edge_color='#B8860B'
        ),
        'Output_Head_1': LayerStyle(
            neuron_fill_color='#90EE90',
            box_around_layer=True,
            box_fill_color='#E6FFE6',
            box_edge_color='#228B22'
        ),
        'Output_Head_2': LayerStyle(
            neuron_fill_color='#FF6B6B',
            box_around_layer=True,
            box_fill_color='#FFE6E6',
            box_edge_color='#DC143C'
        )
    }
)

plot_network(nn5, config=config5,
             save_path=os.path.join(output_dir, "variable_names_multioutput.png"),
             show=False)
print("✓ Saved: variable_names_multioutput.png")

# ==============================================================================
# Example 6: Using layer IDs instead of names
# ==============================================================================
print("\n[Example 6] Using layer IDs for variable names")
print("-" * 70)

nn6 = NeuralNetwork("Network with IDs")

# Store layer IDs
input_id = nn6.add_layer(FullyConnectedLayer(num_neurons=5))
hidden_id = nn6.add_layer(FullyConnectedLayer(num_neurons=8, activation="relu"), 
                          parent_ids=[input_id])
output_id = nn6.add_layer(FullyConnectedLayer(num_neurons=2, activation="sigmoid"), 
                          parent_ids=[hidden_id])

# Use layer IDs in the configuration
config6 = PlotConfig(
    figsize=(12, 6),
    show_title=True,
    show_layer_names=True,
    layer_variable_names={
        input_id: 'Input Features: Sensor Data',
        output_id: 'Binary Classification'
    },
    show_layer_variable_names=True,
    layer_variable_names_fontsize=10,
    layer_variable_names_position='side'
)

plot_network(nn6, config=config6,
             save_path=os.path.join(output_dir, "variable_names_by_id.png"),
             show=False)
print("✓ Saved: variable_names_by_id.png")

print("\n" + "=" * 70)
print("All examples completed!")
print("=" * 70)
print(f"\nGenerated files in: {output_dir}/")
print("\nKey Features Demonstrated:")
print("  1. Variable names positioned on the side (default)")
print("  2. Variable names positioned above layers")
print("  3. Variable names positioned below layers")
print("  4. LaTeX mathematical notation in variable names")
print("  5. Multi-output networks with colored boxes")
print("  6. Using layer IDs instead of layer names")
print("\nConfiguration Options:")
print("  - layer_variable_names: Dict[str, str] - Map layer name/ID to label text")
print("  - show_layer_variable_names: bool - Enable/disable feature")
print("  - layer_variable_names_fontsize: int - Font size for labels")
print("  - layer_variable_names_position: str - 'side', 'above', or 'below'")
print("=" * 70)
