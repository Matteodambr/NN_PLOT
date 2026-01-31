"""
Final verification test for layer grouping positioning.

This test verifies:
1. Group brackets are positioned correctly under their layers
2. Group brackets appear BELOW layer labels (when shown)
3. additional_spacing parameter works correctly
4. Works with both aligned and non-aligned layer labels
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer, VectorInput, VectorOutput
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig, LayerGroup

# Create output directory at project root
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
output_dir = os.path.join(project_root, "test_outputs", "layer_grouping_final")
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("Final Layer Grouping Positioning Verification")
print("="*70)

# Create test network
nn = NeuralNetwork("Positioning Test")
l1_id = nn.add_layer(VectorInput(num_features=8, name="L1"))
l2_id = nn.add_layer(FullyConnectedLayer(10, activation="relu", name="L2"), parent_ids=[l1_id])
l3_id = nn.add_layer(FullyConnectedLayer(10, activation="relu", name="L3"), parent_ids=[l2_id])
l4_id = nn.add_layer(FullyConnectedLayer(8, activation="relu", name="L4"), parent_ids=[l3_id])
l5_id = nn.add_layer(VectorOutput(3, activation="softmax", name="L5"), parent_ids=[l4_id])

print("\nTest 1: With layer names and dimension/activation info...")
config1 = PlotConfig(
    background_color='white',
    show_layer_names=True,
    layer_names_show_dim=True,
    layer_names_show_activation=True,
    figsize=(16, 8),
    layer_groups=[
        LayerGroup(
            layer_ids=["L1", "L2"],
            label="Group A",
            bracket_style='curly',
            bracket_color='blue',
            bracket_linewidth=2.5,
            additional_spacing=0.8
        ),
        LayerGroup(
            layer_ids=["L3", "L4"],
            label="Group B",
            bracket_style='square',
            bracket_color='green',
            bracket_linewidth=2.5,
            additional_spacing=0.8
        )
    ]
)

plot_network(
    nn,
    config=config1,
    title="Test 1: Groups below multi-line layer labels",
    save_path=os.path.join(output_dir, "test1_with_labels.png"),
    show=False,
    dpi=300
)
print("  ✓ Brackets should appear below layer labels with proper spacing")

print("\nTest 2: With aligned bottom layer names...")
config2 = PlotConfig(
    background_color='white',
    show_layer_names=True,
    layer_names_show_dim=True,
    layer_names_align_bottom=True,
    layer_names_bottom_offset=2.5,
    figsize=(16, 8),
    layer_groups=[
        LayerGroup(
            layer_ids=["L1", "L2"],
            label="Group A",
            bracket_style='curly',
            bracket_color='blue',
            bracket_linewidth=2.5,
            additional_spacing=1.2
        ),
        LayerGroup(
            layer_ids=["L3", "L4"],
            label="Group B",
            bracket_style='round',
            bracket_color='green',
            bracket_linewidth=2.5,
            additional_spacing=1.2
        )
    ]
)

plot_network(
    nn,
    config=config2,
    title="Test 2: Groups with aligned bottom layer labels",
    save_path=os.path.join(output_dir, "test2_aligned_bottom.png"),
    show=False,
    dpi=300
)
print("  ✓ All layer labels aligned at bottom, groups below them")

print("\nTest 3: Without layer names (groups below neurons)...")
config3 = PlotConfig(
    background_color='white',
    show_layer_names=False,
    figsize=(16, 8),
    layer_groups=[
        LayerGroup(
            layer_ids=["L1", "L2"],
            label="Group A",
            bracket_style='curly',
            bracket_color='blue',
            bracket_linewidth=2.5,
            y_offset=-1.5
        ),
        LayerGroup(
            layer_ids=["L3", "L4"],
            label="Group B",
            bracket_style='square',
            bracket_color='green',
            bracket_linewidth=2.5,
            y_offset=-1.5
        )
    ]
)

plot_network(
    nn,
    config=config3,
    title="Test 3: Groups directly below neurons (no layer labels)",
    save_path=os.path.join(output_dir, "test3_no_labels.png"),
    show=False,
    dpi=300
)
print("  ✓ Groups positioned using y_offset directly from neurons")

print("\nTest 4: Testing additional_spacing variations...")
config4 = PlotConfig(
    background_color='white',
    show_layer_names=True,
    layer_names_show_dim=True,
    figsize=(16, 8),
    layer_groups=[
        LayerGroup(
            layer_ids=["L1"],
            label="Tight (0.3)",
            bracket_style='curly',
            bracket_color='blue',
            bracket_linewidth=2.0,
            additional_spacing=0.3
        ),
        LayerGroup(
            layer_ids=["L2"],
            label="Default (0.8)",
            bracket_style='curly',
            bracket_color='green',
            bracket_linewidth=2.0,
            additional_spacing=0.8
        ),
        LayerGroup(
            layer_ids=["L3"],
            label="Medium (1.2)",
            bracket_style='curly',
            bracket_color='orange',
            bracket_linewidth=2.0,
            additional_spacing=1.2
        ),
        LayerGroup(
            layer_ids=["L4"],
            label="Large (1.8)",
            bracket_style='curly',
            bracket_color='red',
            bracket_linewidth=2.0,
            additional_spacing=1.8
        )
    ]
)

plot_network(
    nn,
    config=config4,
    title="Test 4: Different additional_spacing values",
    save_path=os.path.join(output_dir, "test4_spacing_variations.png"),
    show=False,
    dpi=300
)
print("  ✓ Each group has different spacing from layer labels")

print("\n" + "="*70)
print("All positioning tests completed!")
print("Check 'test_outputs/layer_grouping_final/' for results")
print("="*70)
print("\nVerify:")
print("  • Groups never overlap with layer labels")
print("  • Groups are positioned correctly under their respective layers")
print("  • additional_spacing parameter controls the gap properly")
print("="*70)
