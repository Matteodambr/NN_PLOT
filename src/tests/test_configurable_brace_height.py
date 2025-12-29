"""
Test script demonstrating configurable brace height.

This tests:
1. Default brace height
2. Tall brace height
3. Short brace height
4. Group bracket with custom height
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig, LayerGroup

# Create output directory
output_dir = os.path.join(os.path.dirname(__file__), "test_outputs", "brace_height")
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("Configurable Brace Height Test Suite")
print("=" * 60)

def test_default_height():
    """Test default brace height."""
    print("\nTesting default brace height (0.15)...")
    
    nn = NeuralNetwork("Default Height")
    nn.add_layer(FullyConnectedLayer(6, name="Input"))
    nn.add_layer(FullyConnectedLayer(8, activation="relu", name="Hidden"))
    nn.add_layer(FullyConnectedLayer(4, activation="softmax", name="Output"))
    
    config = PlotConfig(
        show_layer_names=True,
        layer_names_line_styles=['curly_brace'],
        layer_names_line_color='blue',
        layer_names_line_width=2
    )
    
    plot_network(
        nn,
        config=config,
        title="Default Brace Height (0.15)",
        save_path=os.path.join(output_dir, "01_default_height.png"),
        show=False
    )
    print("  ✓ Default height (0.15)")


def test_tall_height():
    """Test tall brace height."""
    print("\nTesting tall brace height (0.30)...")
    
    nn = NeuralNetwork("Tall Height")
    nn.add_layer(FullyConnectedLayer(6, name="Input"))
    nn.add_layer(FullyConnectedLayer(8, activation="relu", name="Hidden"))
    nn.add_layer(FullyConnectedLayer(4, activation="softmax", name="Output"))
    
    config = PlotConfig(
        show_layer_names=True,
        layer_names_line_styles=['curly_brace'],
        layer_names_line_color='green',
        layer_names_line_width=2,
        layer_names_brace_height=0.30  # Taller braces
    )
    
    plot_network(
        nn,
        config=config,
        title="Tall Brace Height (0.30)",
        save_path=os.path.join(output_dir, "02_tall_height.png"),
        show=False
    )
    print("  ✓ Tall height (0.30)")


def test_short_height():
    """Test short brace height."""
    print("\nTesting short brace height (0.08)...")
    
    nn = NeuralNetwork("Short Height")
    nn.add_layer(FullyConnectedLayer(6, name="Input"))
    nn.add_layer(FullyConnectedLayer(8, activation="relu", name="Hidden"))
    nn.add_layer(FullyConnectedLayer(4, activation="softmax", name="Output"))
    
    config = PlotConfig(
        show_layer_names=True,
        layer_names_line_styles=['curly_brace'],
        layer_names_line_color='red',
        layer_names_line_width=2,
        layer_names_brace_height=0.08  # Shorter braces
    )
    
    plot_network(
        nn,
        config=config,
        title="Short Brace Height (0.08)",
        save_path=os.path.join(output_dir, "03_short_height.png"),
        show=False
    )
    print("  ✓ Short height (0.08)")


def test_group_bracket_heights():
    """Test group brackets with different heights."""
    print("\nTesting group brackets with different heights...")
    
    nn = NeuralNetwork("Group Heights")
    nn.add_layer(FullyConnectedLayer(6, name="L1"))
    nn.add_layer(FullyConnectedLayer(8, activation="relu", name="L2"))
    nn.add_layer(FullyConnectedLayer(8, activation="relu", name="L3"))
    nn.add_layer(FullyConnectedLayer(4, activation="softmax", name="L4"))
    
    config = PlotConfig(
        show_layer_names=True,
        layer_groups=[
            LayerGroup(
                layer_ids=["L1", "L2"],
                label="Short Bracket (0.2)",
                bracket_style='curly',
                bracket_color='purple',
                bracket_linewidth=2.0,
                bracket_height=0.2,
                label_fontsize=11,
                label_color='purple'
            ),
            LayerGroup(
                layer_ids=["L3", "L4"],
                label="Tall Bracket (0.6)",
                bracket_style='curly',
                bracket_color='orange',
                bracket_linewidth=2.0,
                bracket_height=0.6,
                label_fontsize=11,
                label_color='orange'
            )
        ]
    )
    
    plot_network(
        nn,
        config=config,
        title="Group Brackets with Different Heights",
        save_path=os.path.join(output_dir, "04_group_heights.png"),
        show=False
    )
    print("  ✓ Group brackets with different heights")


if __name__ == "__main__":
    test_default_height()
    test_tall_height()
    test_short_height()
    test_group_bracket_heights()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("Check 'test_outputs/brace_height' for results.")
    print("=" * 60)
