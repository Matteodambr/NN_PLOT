"""
Demonstration of layer boxes feature - drawing rounded boxes around specific layers.

This example shows how to:
1. Add rounded boxes around specific layers (e.g., output heads)
2. Customize box appearance (color, edge width, corner radius)
3. Adjust figure width for better visualization
"""

import sys
sys.path.insert(0, 'src')
from NN_PLOTTING_UTILITIES import (
    NeuralNetwork,
    FullyConnectedLayer,
    PlotConfig,
    LayerStyle,
    NetworkPlotter
)
from NN_DEFINITION_UTILITIES import VectorInput

print("=" * 70)
print("Layer Boxes Demonstration")
print("=" * 70)

# ============================================================================
# Example 1: Simple branching network with boxes around output heads
# ============================================================================
print("\n[Example 1] Branching network with boxed output heads")
print("-" * 70)

network1 = NeuralNetwork("Boxed_Heads_Network")

# Input layer
input_layer = VectorInput(num_features=6, name="Input")
input_id = network1.add_layer(input_layer)

# Hidden layers
hidden1 = FullyConnectedLayer(num_neurons=300, name="Hidden_1")
hidden1_id = network1.add_layer(hidden1, parent_ids=[input_id])

hidden2 = FullyConnectedLayer(num_neurons=300, name="Hidden_2")
hidden2_id = network1.add_layer(hidden2, parent_ids=[hidden1_id])

hidden3 = FullyConnectedLayer(num_neurons=300, name="Hidden_3")
hidden3_id = network1.add_layer(hidden3, parent_ids=[hidden2_id])

# Two output heads with different box styles
output1 = FullyConnectedLayer(num_neurons=7, name="Head_A")
output1_id = network1.add_layer(output1, parent_ids=[hidden3_id])

output2 = FullyConnectedLayer(num_neurons=7, name="Head_B")
output2_id = network1.add_layer(output2, parent_ids=[hidden3_id])

# Configure with boxes around the output heads
config1 = PlotConfig(
    figsize=(16, 8),  # Wider figure for better visibility
    show_layer_names=True,
    show_neuron_labels=False,
    background_color='white',
    layer_styles={
        'Head_A': LayerStyle(
            neuron_fill_color='lightcoral',
            neuron_edge_color='darkred',
            box_around_layer=True,
            box_fill_color='#FFE6E6',  # Light red
            box_edge_color='darkred',
            box_edge_width=2.5,
            box_padding=0.6,
            box_corner_radius=0.4
        ),
        'Head_B': LayerStyle(
            neuron_fill_color='lightgreen',
            neuron_edge_color='darkgreen',
            box_around_layer=True,
            box_fill_color='#E6FFE6',  # Light green
            box_edge_color='darkgreen',
            box_edge_width=2.5,
            box_padding=0.6,
            box_corner_radius=0.4
        )
    }
)

plotter1 = NetworkPlotter(config1)
plotter1.plot_network(
    network1,
    title="Neural Network with Boxed Output Heads",
    save_path="examples/demo_boxes_example1.png",
    show=False,
    dpi=300
)
print("✓ Saved: examples/demo_boxes_example1.png")

# ============================================================================
# Example 2: Multiple box styles
# ============================================================================
print("\n[Example 2] Different box styles")
print("-" * 70)

network2 = NeuralNetwork("Multi_Style_Network")

# Build a simple linear network
input2 = VectorInput(num_features=4, name="Input")
input2_id = network2.add_layer(input2)

hidden_a = FullyConnectedLayer(num_neurons=8, name="Encoder")
hidden_a_id = network2.add_layer(hidden_a, parent_ids=[input2_id])

hidden_b = FullyConnectedLayer(num_neurons=6, name="Latent")
hidden_b_id = network2.add_layer(hidden_b, parent_ids=[hidden_a_id])

hidden_c = FullyConnectedLayer(num_neurons=8, name="Decoder")
hidden_c_id = network2.add_layer(hidden_c, parent_ids=[hidden_b_id])

output_final = FullyConnectedLayer(num_neurons=4, name="Output")
output_final_id = network2.add_layer(output_final, parent_ids=[hidden_c_id])

# Different box styles for different parts of the network
config2 = PlotConfig(
    figsize=(14, 6),  # Custom width
    show_layer_names=True,
    show_neuron_labels=True,
    background_color='white',
    layer_styles={
        'Encoder': LayerStyle(
            neuron_fill_color='skyblue',
            box_around_layer=True,
            box_fill_color=None,  # No fill, just border
            box_edge_color='blue',
            box_edge_width=3.0,
            box_padding=0.5,
            box_corner_radius=0.5
        ),
        'Latent': LayerStyle(
            neuron_fill_color='gold',
            box_around_layer=True,
            box_fill_color='#FFFACD',  # Light yellow
            box_edge_color='orange',
            box_edge_width=2.0,
            box_padding=0.8,
            box_corner_radius=0.3
        ),
        'Decoder': LayerStyle(
            neuron_fill_color='lightcoral',
            box_around_layer=True,
            box_fill_color=None,  # No fill
            box_edge_color='red',
            box_edge_width=3.0,
            box_padding=0.5,
            box_corner_radius=0.5
        )
    }
)

plotter2 = NetworkPlotter(config2)
plotter2.plot_network(
    network2,
    title="Network with Different Box Styles",
    save_path="examples/demo_boxes_example2.png",
    show=False,
    dpi=300
)
print("✓ Saved: examples/demo_boxes_example2.png")

# ============================================================================
# Example 3: Wide figure with collapsed layers and boxes
# ============================================================================
print("\n[Example 3] Wide figure with collapsed layers and boxes")
print("-" * 70)

network3 = NeuralNetwork("Wide_Boxed_Network")

# Create a network similar to CEAS2025 but with boxes
inp3 = VectorInput(num_features=10, name="Input")
inp3_id = network3.add_layer(inp3)

h1 = FullyConnectedLayer(num_neurons=200, name="Hidden_1")
h1_id = network3.add_layer(h1, parent_ids=[inp3_id])

h2 = FullyConnectedLayer(num_neurons=200, name="Hidden_2")
h2_id = network3.add_layer(h2, parent_ids=[h1_id])

# Three output heads with boxes
out1 = FullyConnectedLayer(num_neurons=50, name="Classification")
network3.add_layer(out1, parent_ids=[h2_id])

out2 = FullyConnectedLayer(num_neurons=50, name="Regression")
network3.add_layer(out2, parent_ids=[h2_id])

out3 = FullyConnectedLayer(num_neurons=50, name="Embedding")
network3.add_layer(out3, parent_ids=[h2_id])

config3 = PlotConfig(
    figsize=(18, 10),  # Extra wide for complex network
    show_layer_names=True,
    show_neuron_labels=False,
    background_color='white',
    max_neurons_per_layer=20,  # Collapse large layers
    layer_styles={
        'Classification': LayerStyle(
            neuron_fill_color='#FFB6C1',  # Light pink
            box_around_layer=True,
            box_fill_color='#FFE4E1',
            box_edge_color='#FF1493',  # Deep pink
            box_edge_width=2.0,
            box_padding=0.7,
            box_corner_radius=0.4
        ),
        'Regression': LayerStyle(
            neuron_fill_color='#ADD8E6',  # Light blue
            box_around_layer=True,
            box_fill_color='#E0F2FF',
            box_edge_color='#0000CD',  # Medium blue
            box_edge_width=2.0,
            box_padding=0.7,
            box_corner_radius=0.4
        ),
        'Embedding': LayerStyle(
            neuron_fill_color='#98FB98',  # Pale green
            box_around_layer=True,
            box_fill_color='#E6FFE6',
            box_edge_color='#228B22',  # Forest green
            box_edge_width=2.0,
            box_padding=0.7,
            box_corner_radius=0.4
        )
    }
)

plotter3 = NetworkPlotter(config3)
plotter3.plot_network(
    network3,
    title="Wide Network with Collapsed Layers and Boxed Heads",
    save_path="examples/demo_boxes_example3.png",
    show=False,
    dpi=300
)
print("✓ Saved: examples/demo_boxes_example3.png")

# ============================================================================
# Example 4: Minimal boxes - just borders
# ============================================================================
print("\n[Example 4] Minimal box style - borders only")
print("-" * 70)

network4 = NeuralNetwork("Minimal_Boxes")

inp4 = VectorInput(num_features=3, name="Input")
inp4_id = network4.add_layer(inp4)

h4 = FullyConnectedLayer(num_neurons=5, name="Hidden")
h4_id = network4.add_layer(h4, parent_ids=[inp4_id])

out4 = FullyConnectedLayer(num_neurons=2, name="Output")
network4.add_layer(out4, parent_ids=[h4_id])

config4 = PlotConfig(
    figsize=(10, 6),
    show_layer_names=True,
    show_neuron_labels=True,
    background_color='white',
    layer_styles={
        'Input': LayerStyle(
            box_around_layer=True,
            box_fill_color=None,  # No fill
            box_edge_color='black',
            box_edge_width=1.5,
            box_padding=0.4,
            box_corner_radius=0.2
        ),
        'Output': LayerStyle(
            neuron_fill_color='lightcoral',
            box_around_layer=True,
            box_fill_color=None,
            box_edge_color='black',
            box_edge_width=1.5,
            box_padding=0.4,
            box_corner_radius=0.2
        )
    }
)

plotter4 = NetworkPlotter(config4)
plotter4.plot_network(
    network4,
    title="Minimal Box Style - Borders Only",
    save_path="examples/demo_boxes_example4.png",
    show=False,
    dpi=300
)
print("✓ Saved: examples/demo_boxes_example4.png")

print("\n" + "=" * 70)
print("All examples completed!")
print("=" * 70)
print("\nGenerated files:")
print("  - examples/demo_boxes_example1.png (Branching network with colored boxes)")
print("  - examples/demo_boxes_example2.png (Different box styles)")
print("  - examples/demo_boxes_example3.png (Wide figure with collapsed layers)")
print("  - examples/demo_boxes_example4.png (Minimal border-only boxes)")
print("\nKey features demonstrated:")
print("  ✓ Rounded boxes around specific layers")
print("  ✓ Customizable box colors (fill and edge)")
print("  ✓ Adjustable edge width and corner radius")
print("  ✓ Variable figure width (figsize parameter)")
print("  ✓ Boxes work with both linear and branching networks")
print("  ✓ Boxes work with collapsed layers")
