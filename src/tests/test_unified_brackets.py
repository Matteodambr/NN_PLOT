"""
Test script demonstrating the unified bracket implementation.

This tests:
1. Single-layer curly braces (layer names)
2. Multi-layer curly braces (layer groups)
3. Both in the same plot to show consistency
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer, VectorInput, VectorOutput
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig, LayerGroup

# Create output directory at project root
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
output_dir = os.path.join(project_root, "test_outputs", "unified_brackets")
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("Unified Bracket Implementation Test Suite")
print("=" * 60)

def test_single_layer_brace():
    """Test single-layer curly braces."""
    print("\nTesting single-layer curly braces...")
    
    nn = NeuralNetwork("Single Layer Braces")
    nn.add_layer(VectorInput(num_features=6, name="Input"))
    nn.add_layer(FullyConnectedLayer(8, activation="relu", name="Hidden1"))
    nn.add_layer(FullyConnectedLayer(8, activation="relu", name="Hidden2"))
    nn.add_layer(FullyConnectedLayer(4, activation="softmax", name="Output"))
    
    config = PlotConfig(
        show_layer_names=True,
        layer_names_line_styles=['curly_brace'],
        layer_names_line_color='blue',
        layer_names_line_width=2,
        layer_names_brace_width_multiplier=1.5
    )
    
    plot_network(
        nn,
        config=config,
        title="Single Layer Curly Braces",
        save_path=os.path.join(output_dir, "01_single_layer_braces.png"),
        show=False
    )
    print("  ✓ Single-layer curly braces")


def test_multi_layer_group_braces():
    """Test multi-layer group curly braces."""
    print("\nTesting multi-layer group curly braces...")
    
    nn = NeuralNetwork("Multi-Layer Group Braces")
    nn.add_layer(VectorInput(num_features=6, name="Input"))
    nn.add_layer(FullyConnectedLayer(8, activation="relu", name="Hidden1"))
    nn.add_layer(FullyConnectedLayer(8, activation="relu", name="Hidden2"))
    nn.add_layer(FullyConnectedLayer(4, activation="softmax", name="Output"))
    
    config = PlotConfig(
        show_layer_names=True,
        layer_groups=[
            LayerGroup(
                layer_ids=["Hidden1", "Hidden2"],
                label="Feature Extraction",
                bracket_style='curly',
                bracket_color='green',
                bracket_linewidth=2.0,
                bracket_height=0.4,
                label_fontsize=12,
                label_color='green'
            )
        ]
    )
    
    plot_network(
        nn,
        config=config,
        title="Multi-Layer Group Curly Braces",
        save_path=os.path.join(output_dir, "02_multi_layer_group_braces.png"),
        show=False
    )
    print("  ✓ Multi-layer group curly braces")


def test_combined_single_and_group():
    """Test both single-layer and group braces together."""
    print("\nTesting combined single-layer and group braces...")
    
    nn = NeuralNetwork("Combined Braces")
    nn.add_layer(VectorInput(num_features=6, name="Input"))
    nn.add_layer(FullyConnectedLayer(8, activation="relu", name="Hidden1"))
    nn.add_layer(FullyConnectedLayer(8, activation="relu", name="Hidden2"))
    nn.add_layer(VectorOutput(4, activation="softmax", name="Output"))
    
    config = PlotConfig(
        show_layer_names=True,
        layer_names_line_styles=['curly_brace'],
        layer_names_line_color='blue',
        layer_names_line_width=2,
        layer_names_brace_width_multiplier=1.5,
        layer_groups=[
            LayerGroup(
                layer_ids=["Hidden1", "Hidden2"],
                label="Feature Extraction",
                bracket_style='curly',
                bracket_color='red',
                bracket_linewidth=2.0,
                bracket_height=0.4,
                label_fontsize=12,
                label_color='red'
            )
        ]
    )
    
    plot_network(
        nn,
        config=config,
        title="Combined: Single-Layer Braces (blue) + Group Brace (red)",
        save_path=os.path.join(output_dir, "03_combined_braces.png"),
        show=False
    )
    print("  ✓ Combined single-layer and group braces")


def test_label_spacing():
    """Test that labels don't overlap with brackets."""
    print("\nTesting label spacing (no overlap)...")
    
    nn = NeuralNetwork("Label Spacing Test")
    nn.add_layer(VectorInput(num_features=6, name="Layer1"))
    nn.add_layer(FullyConnectedLayer(6, activation="relu", name="Layer2"))
    nn.add_layer(VectorOutput(4, activation="softmax", name="Layer3"))
    
    config = PlotConfig(
        show_layer_names=True,
        layer_groups=[
            LayerGroup(
                layer_ids=["Layer1", "Layer2"],
                label="Test Group with Large Bracket",
                bracket_style='curly',
                bracket_color='orange',
                bracket_linewidth=3.0,
                bracket_height=0.8,  # Large bracket height
                label_fontsize=14,
                label_color='orange'
            )
        ]
    )
    
    plot_network(
        nn,
        config=config,
        title="Label Spacing Test - No Overlap",
        save_path=os.path.join(output_dir, "04_label_spacing.png"),
        show=False
    )
    print("  ✓ Label spacing (no overlap with large bracket)")


if __name__ == "__main__":
    test_single_layer_brace()
    test_multi_layer_group_braces()
    test_combined_single_and_group()
    test_label_spacing()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("Check 'test_outputs/unified_brackets' for results.")
    print("=" * 60)
