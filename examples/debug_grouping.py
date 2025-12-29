"""Quick test to debug layer grouping positioning"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig, LayerGroup

nn = NeuralNetwork("Debug Test")

l1_id = nn.add_layer(FullyConnectedLayer(5, name="Layer_1"))
l2_id = nn.add_layer(FullyConnectedLayer(5, activation="relu", name="Layer_2"), parent_ids=[l1_id])
l3_id = nn.add_layer(FullyConnectedLayer(5, activation="relu", name="Layer_3"), parent_ids=[l2_id])
l4_id = nn.add_layer(FullyConnectedLayer(5, activation="relu", name="Layer_4"), parent_ids=[l3_id])
l5_id = nn.add_layer(FullyConnectedLayer(3, activation="softmax", name="Layer_5"), parent_ids=[l4_id])

print(f"Layer IDs: {l1_id}, {l2_id}, {l3_id}, {l4_id}, {l5_id}")

config = PlotConfig(
    background_color='white',
    show_layer_names=True,
    figsize=(16, 7),
    layer_groups=[
        LayerGroup(
            layer_ids=["Layer_1", "Layer_2"],
            label="Group A (Layers 1-2)",
            bracket_style='curly',
            bracket_color='blue',
            bracket_linewidth=2.5,
            label_fontsize=12,
            y_offset=-1.8
        ),
        LayerGroup(
            layer_ids=["Layer_3", "Layer_4"],
            label="Group B (Layers 3-4)",
            bracket_style='square',
            bracket_color='green',
            bracket_linewidth=2.5,
            label_fontsize=12,
            y_offset=-1.8
        ),
        LayerGroup(
            layer_ids=["Layer_5"],
            label="Group C (Layer 5)",
            bracket_style='round',
            bracket_color='red',
            bracket_linewidth=2.5,
            label_fontsize=12,
            y_offset=-1.8
        )
    ]
)

plot_network(
    nn,
    config=config,
    title="Debug: Layer Grouping Positioning",
    save_path="test_outputs/debug_grouping.png",
    show=False,
    dpi=300
)

print("✅ Created: test_outputs/debug_grouping.png")
print("Check if brackets are positioned correctly under their respective layers")
