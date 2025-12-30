"""
Verification that all group brackets are at the same height.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer, VectorInput
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig, LayerGroup

print("="*70)
print("Verifying: All group brackets at SAME HEIGHT")
print("="*70)

nn = NeuralNetwork("Same Height Test")

# Create layers with different sizes
l1_id = nn.add_layer(VectorInput(num_features=5, name="Small_1"))
l2_id = nn.add_layer(FullyConnectedLayer(12, activation="relu", name="Large_1"), parent_ids=[l1_id])
l3_id = nn.add_layer(FullyConnectedLayer(8, activation="relu", name="Medium_1"), parent_ids=[l2_id])
l4_id = nn.add_layer(FullyConnectedLayer(15, activation="relu", name="VeryLarge"), parent_ids=[l3_id])
l5_id = nn.add_layer(FullyConnectedLayer(4, activation="softmax", name="Tiny"), parent_ids=[l4_id])

config = PlotConfig(
    background_color='white',
    show_layer_names=True,
    layer_names_show_dim=True,
    layer_names_show_activation=True,
    figsize=(18, 10),
    layer_groups=[
        LayerGroup(
            layer_ids=["Small_1"],
            label="Group 1",
            bracket_style='curly',
            bracket_color='blue',
            bracket_linewidth=3.0,
            label_fontsize=14,
            label_color='blue',
            additional_spacing=1.0
        ),
        LayerGroup(
            layer_ids=["Large_1", "Medium_1"],
            label="Group 2",
            bracket_style='square',
            bracket_color='green',
            bracket_linewidth=3.0,
            label_fontsize=14,
            label_color='green',
            additional_spacing=1.0
        ),
        LayerGroup(
            layer_ids=["VeryLarge"],
            label="Group 3",
            bracket_style='round',
            bracket_color='orange',
            bracket_linewidth=3.0,
            label_fontsize=14,
            label_color='orange',
            additional_spacing=1.0
        ),
        LayerGroup(
            layer_ids=["Tiny"],
            label="Group 4",
            bracket_style='straight',
            bracket_color='red',
            bracket_linewidth=3.0,
            label_fontsize=14,
            label_color='red',
            additional_spacing=1.0
        )
    ]
)

plot_network(
    nn,
    config=config,
    title="Verification: All Brackets at Same Height (despite different layer sizes)",
    save_path="test_outputs/verify_same_height.png",
    show=False,
    dpi=300
)

print("\n✅ Created: test_outputs/verify_same_height.png")
print("\nVERIFY IN IMAGE:")
print("  ✓ All 4 brackets should be at EXACTLY the same y-position")
print("  ✓ Brackets correctly span their respective layers (x-position)")
print("  ✓ All labels appear at the same height below brackets")
print("  ✓ Different bracket styles are clearly visible")
print("="*70)
