"""Test layer grouping with layer names enabled"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer, VectorInput
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig, LayerGroup

print("Testing layer grouping with layer names...")

nn = NeuralNetwork("Test Network")

l1_id = nn.add_layer(VectorInput(num_features=8, name="Input"))
l2_id = nn.add_layer(FullyConnectedLayer(10, activation="relu", name="Hidden1"), parent_ids=[l1_id])
l3_id = nn.add_layer(FullyConnectedLayer(10, activation="relu", name="Hidden2"), parent_ids=[l2_id])
l4_id = nn.add_layer(FullyConnectedLayer(8, activation="relu", name="Hidden3"), parent_ids=[l3_id])
l5_id = nn.add_layer(FullyConnectedLayer(3, activation="softmax", name="Output"), parent_ids=[l4_id])

# Test 1: With layer names shown
config_with_names = PlotConfig(
    background_color='white',
    show_layer_names=True,
    layer_names_show_dim=True,
    layer_names_show_activation=True,
    figsize=(16, 8),
    layer_groups=[
        LayerGroup(
            layer_ids=["Input", "Hidden1"],
            label="Stage 1",
            bracket_style='curly',
            bracket_color='blue',
            bracket_linewidth=2.5,
            label_fontsize=13,
            label_color='blue',
            additional_spacing=0.5
        ),
        LayerGroup(
            layer_ids=["Hidden2", "Hidden3"],
            label="Stage 2",
            bracket_style='curly',
            bracket_color='green',
            bracket_linewidth=2.5,
            label_fontsize=13,
            label_color='green',
            additional_spacing=0.5
        ),
        LayerGroup(
            layer_ids=["Output"],
            label="Stage 3",
            bracket_style='square',
            bracket_color='red',
            bracket_linewidth=2.5,
            label_fontsize=13,
            label_color='red',
            additional_spacing=0.5
        )
    ]
)

plot_network(
    nn,
    config=config_with_names,
    title="Layer Grouping WITH Layer Names (brackets should be below layer labels)",
    save_path="test_outputs/debug_grouping_with_names.png",
    show=False,
    dpi=300
)
print("✅ Created: test_outputs/debug_grouping_with_names.png")

# Test 2: Without layer names
config_without_names = PlotConfig(
    background_color='white',
    show_layer_names=False,
    figsize=(16, 8),
    layer_groups=[
        LayerGroup(
            layer_ids=["Input", "Hidden1"],
            label="Stage 1",
            bracket_style='curly',
            bracket_color='blue',
            bracket_linewidth=2.5,
            label_fontsize=13,
            label_color='blue'
        ),
        LayerGroup(
            layer_ids=["Hidden2", "Hidden3"],
            label="Stage 2",
            bracket_style='curly',
            bracket_color='green',
            bracket_linewidth=2.5,
            label_fontsize=13,
            label_color='green'
        ),
        LayerGroup(
            layer_ids=["Output"],
            label="Stage 3",
            bracket_style='square',
            bracket_color='red',
            bracket_linewidth=2.5,
            label_fontsize=13,
            label_color='red'
        )
    ]
)

plot_network(
    nn,
    config=config_without_names,
    title="Layer Grouping WITHOUT Layer Names (brackets should be below neurons)",
    save_path="test_outputs/debug_grouping_without_names.png",
    show=False,
    dpi=300
)
print("✅ Created: test_outputs/debug_grouping_without_names.png")

# Test 3: With aligned bottom layer names
config_aligned_bottom = PlotConfig(
    background_color='white',
    show_layer_names=True,
    layer_names_show_dim=True,
    layer_names_show_activation=True,
    layer_names_align_bottom=True,
    layer_names_bottom_offset=2.5,
    figsize=(16, 8),
    layer_groups=[
        LayerGroup(
            layer_ids=["Input", "Hidden1"],
            label="Stage 1",
            bracket_style='curly',
            bracket_color='blue',
            bracket_linewidth=2.5,
            label_fontsize=13,
            label_color='blue',
            additional_spacing=0.8
        ),
        LayerGroup(
            layer_ids=["Hidden2", "Hidden3"],
            label="Stage 2",
            bracket_style='curly',
            bracket_color='green',
            bracket_linewidth=2.5,
            label_fontsize=13,
            label_color='green',
            additional_spacing=0.8
        ),
        LayerGroup(
            layer_ids=["Output"],
            label="Stage 3",
            bracket_style='square',
            bracket_color='red',
            bracket_linewidth=2.5,
            label_fontsize=13,
            label_color='red',
            additional_spacing=0.8
        )
    ]
)

plot_network(
    nn,
    config=config_aligned_bottom,
    title="Layer Grouping with ALIGNED BOTTOM layer names",
    save_path="test_outputs/debug_grouping_aligned_bottom.png",
    show=False,
    dpi=300
)
print("✅ Created: test_outputs/debug_grouping_aligned_bottom.png")

print("\nAll tests complete! Check the output images to verify:")
print("  1. Brackets are positioned correctly below their respective layers")
print("  2. Brackets appear below layer labels (not overlapping)")
print("  3. additional_spacing parameter controls the gap")
