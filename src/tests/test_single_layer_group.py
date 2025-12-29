"""
Test script to verify single-layer grouping works correctly.
This addresses issue #2 - single layer groups should span properly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use('Agg')

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig, LayerGroup

output_dir = os.path.join(os.path.dirname(__file__), "test_outputs", "single_layer_group")
os.makedirs(output_dir, exist_ok=True)

print("Testing single-layer group bracket...")

nn = NeuralNetwork("Single Layer Group Test")
l1_id = nn.add_layer(FullyConnectedLayer(8, name="Input"))
l2_id = nn.add_layer(FullyConnectedLayer(10, activation="relu", name="Processing"), parent_ids=[l1_id])
l3_id = nn.add_layer(FullyConnectedLayer(6, activation="softmax", name="Output"), parent_ids=[l2_id])

# Test with single layer in a group - should span the full width of that layer
config = PlotConfig(
    background_color='white',
    show_layer_names=True,
    layer_groups=[
        LayerGroup(
            layer_ids=["Processing"],  # Single layer
            label="Single Layer Group",
            bracket_style='curly',
            bracket_color='#1976D2',
            bracket_linewidth=2.0,
            label_fontsize=12,
            label_color='#1976D2'
        )
    ]
)

plot_network(
    nn,
    config=config,
    title="Single Layer Group Test",
    save_path=os.path.join(output_dir, "single_layer_curly.png"),
    show=False
)
print("  ✓ Single layer group with curly bracket")

# Test with multiple single-layer groups
config2 = PlotConfig(
    background_color='white',
    show_layer_names=True,
    layer_groups=[
        LayerGroup(
            layer_ids=["Input"],
            label="Input Layer",
            bracket_style='square',
            bracket_color='#388E3C',
            bracket_linewidth=2.0
        ),
        LayerGroup(
            layer_ids=["Output"],
            label="Output Layer",
            bracket_style='round',
            bracket_color='#D32F2F',
            bracket_linewidth=2.0
        )
    ]
)

plot_network(
    nn,
    config=config2,
    title="Multiple Single Layer Groups",
    save_path=os.path.join(output_dir, "multiple_single_layers.png"),
    show=False
)
print("  ✓ Multiple single layer groups")

print("\n✓ All single-layer group tests passed!")
print(f"Check '{output_dir}' for results.")
