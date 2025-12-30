"""
Demonstration of:
1. Text labels positioned outside layer boxes
2. Layer spacing multiplier for wider networks
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
print("Text Labels Outside Boxes & Spacing Multiplier Demo")
print("=" * 70)

# ============================================================================
# Example 1: Text labels with boxes (default spacing)
# ============================================================================
print("\n[Example 1] Text labels positioned outside boxes (default spacing)")
print("-" * 70)

network1 = NeuralNetwork("Labels_Outside_Boxes")

# Create network with text labels
input_layer = VectorInput(
    num_features=3,
    name="Input",
    neuron_labels=[r"$x_1$", r"$x_2$", r"$x_3$"],
    label_position="left"
)
input_id = network1.add_layer(input_layer)

hidden_layer = FullyConnectedLayer(
    num_neurons=4,
    name="Hidden",
    neuron_labels=[r"$h_1$", r"$h_2$", r"$h_3$", r"$h_4$"]
)
hidden_id = network1.add_layer(hidden_layer, parent_ids=[input_id])

output_layer = FullyConnectedLayer(
    num_neurons=2,
    name="Output",
    neuron_labels=[r"$y_1$", r"$y_2$"],
    label_position="right"
)
network1.add_layer(output_layer, parent_ids=[hidden_id])

# Configure with boxes around input and output
config1 = PlotConfig(
    figsize=(12, 6),
    background_color='white',
    show_layer_names=True,
    show_neuron_labels=False,
    show_neuron_text_labels=True,
    layer_styles={
        'Input': LayerStyle(
            neuron_fill_color='lightblue',
            box_around_layer=True,
            box_fill_color='#E0F2FF',
            box_edge_color='darkblue',
            box_edge_width=2.0,
            box_padding=0.6
        ),
        'Output': LayerStyle(
            neuron_fill_color='lightcoral',
            box_around_layer=True,
            box_fill_color='#FFE6E6',
            box_edge_color='darkred',
            box_edge_width=2.0,
            box_padding=0.6
        )
    }
)

plotter1 = NetworkPlotter(config1)
plotter1.plot_network(
    network1,
    title="Text Labels Positioned Outside Boxes (Default Spacing)",
    save_path="examples/demo_spacing_example1.png",
    show=False
)
print("✓ Saved: examples/demo_spacing_example1.png")

# ============================================================================
# Example 2: Same network with 1.5x spacing multiplier
# ============================================================================
print("\n[Example 2] Same network with 1.5x spacing multiplier (50% wider)")
print("-" * 70)

config2 = PlotConfig(
    figsize=(14, 6),  # Slightly larger figure for wider content
    background_color='white',
    show_layer_names=True,
    show_neuron_labels=False,
    show_neuron_text_labels=True,
    layer_spacing_multiplier=1.5,  # Make network 50% wider
    layer_styles={
        'Input': LayerStyle(
            neuron_fill_color='lightblue',
            box_around_layer=True,
            box_fill_color='#E0F2FF',
            box_edge_color='darkblue',
            box_edge_width=2.0,
            box_padding=0.6
        ),
        'Output': LayerStyle(
            neuron_fill_color='lightcoral',
            box_around_layer=True,
            box_fill_color='#FFE6E6',
            box_edge_color='darkred',
            box_edge_width=2.0,
            box_padding=0.6
        )
    }
)

plotter2 = NetworkPlotter(config2)
plotter2.plot_network(
    network1,
    title="Same Network with 1.5x Spacing Multiplier",
    save_path="examples/demo_spacing_example2.png",
    show=False
)
print("✓ Saved: examples/demo_spacing_example2.png")

# ============================================================================
# Example 3: Cramped branching network (default)
# ============================================================================
print("\n[Example 3] Branching network - default spacing")
print("-" * 70)

network3 = NeuralNetwork("Branching_Default")

inp = VectorInput(
    num_features=4,
    name="Input",
    neuron_labels=[r"$i_1$", r"$i_2$", r"$i_3$", r"$i_4$"],
    label_position="left"
)
inp_id = network3.add_layer(inp)

h1 = FullyConnectedLayer(num_neurons=6, name="Hidden")
h1_id = network3.add_layer(h1, parent_ids=[inp_id])

# Two output heads with labels
out1 = FullyConnectedLayer(
    num_neurons=3,
    name="Head_A",
    neuron_labels=[r"$a_1$", r"$a_2$", r"$a_3$"],
    label_position="right"
)
network3.add_layer(out1, parent_ids=[h1_id])

out2 = FullyConnectedLayer(
    num_neurons=3,
    name="Head_B",
    neuron_labels=[r"$b_1$", r"$b_2$", r"$b_3$"],
    label_position="right"
)
network3.add_layer(out2, parent_ids=[h1_id])

config3 = PlotConfig(
    figsize=(12, 7),
    background_color='white',
    show_layer_names=True,
    show_neuron_labels=False,
    show_neuron_text_labels=True,
    layer_styles={
        'Head_A': LayerStyle(
            neuron_fill_color='#FFD700',
            box_around_layer=True,
            box_fill_color='#FFFACD',
            box_edge_color='#B8860B',
            box_edge_width=2.0,
            box_padding=0.7
        ),
        'Head_B': LayerStyle(
            neuron_fill_color='#87CEEB',
            box_around_layer=True,
            box_fill_color='#E0F2FF',
            box_edge_color='#4682B4',
            box_edge_width=2.0,
            box_padding=0.7
        )
    }
)

plotter3 = NetworkPlotter(config3)
plotter3.plot_network(
    network3,
    title="Branching Network - Default Spacing",
    save_path="examples/demo_spacing_example3.png",
    show=False
)
print("✓ Saved: examples/demo_spacing_example3.png")

# ============================================================================
# Example 4: Same branching network with 2.0x spacing multiplier
# ============================================================================
print("\n[Example 4] Same branching network with 2.0x spacing (100% wider)")
print("-" * 70)

config4 = PlotConfig(
    figsize=(16, 7),  # Wider figure
    background_color='white',
    show_layer_names=True,
    show_neuron_labels=False,
    show_neuron_text_labels=True,
    layer_spacing_multiplier=2.0,  # Double the spacing
    layer_styles={
        'Head_A': LayerStyle(
            neuron_fill_color='#FFD700',
            box_around_layer=True,
            box_fill_color='#FFFACD',
            box_edge_color='#B8860B',
            box_edge_width=2.0,
            box_padding=0.7
        ),
        'Head_B': LayerStyle(
            neuron_fill_color='#87CEEB',
            box_around_layer=True,
            box_fill_color='#E0F2FF',
            box_edge_color='#4682B4',
            box_edge_width=2.0,
            box_padding=0.7
        )
    }
)

plotter4 = NetworkPlotter(config4)
plotter4.plot_network(
    network3,
    title="Same Network with 2.0x Spacing Multiplier",
    save_path="examples/demo_spacing_example4.png",
    show=False
)
print("✓ Saved: examples/demo_spacing_example4.png")

# ============================================================================
# Example 5: Comparison of different spacing multipliers
# ============================================================================
print("\n[Example 5] Multiple spacing multipliers for comparison")
print("-" * 70)

# Simple network for comparison
network5 = NeuralNetwork("Spacing_Compare")
i = network5.add_layer(VectorInput(num_features=3, name="A"))
h = network5.add_layer(FullyConnectedLayer(num_neurons=4, name="B"))
network5.add_layer(FullyConnectedLayer(num_neurons=2, name="C"))

for mult, desc in [(1.0, "default"), (1.5, "1.5x"), (2.0, "2x"), (2.5, "2.5x")]:
    config = PlotConfig(
        figsize=(8 + mult * 2, 5),
        background_color='white',
        show_layer_names=True,
        layer_spacing_multiplier=mult
    )
    plotter = NetworkPlotter(config)
    plotter.plot_network(
        network5,
        title=f"Spacing Multiplier: {mult}x",
        save_path=f"examples/demo_spacing_compare_{desc}.png",
        show=False
    )
    print(f"✓ Saved: examples/demo_spacing_compare_{desc}.png")

print("\n" + "=" * 70)
print("All examples completed!")
print("=" * 70)
print("\nGenerated files:")
print("  - demo_spacing_example1.png (labels outside boxes, default spacing)")
print("  - demo_spacing_example2.png (labels outside boxes, 1.5x spacing)")
print("  - demo_spacing_example3.png (branching network, default spacing)")
print("  - demo_spacing_example4.png (branching network, 2.0x spacing)")
print("  - demo_spacing_compare_*.png (comparison of multipliers)")
print("\nKey features demonstrated:")
print("  ✓ Text labels automatically positioned outside boxes")
print("  ✓ Layer spacing multiplier for wider networks")
print("  ✓ Works with both linear and branching networks")
print("  ✓ Maintains relative spacing while scaling overall width")
