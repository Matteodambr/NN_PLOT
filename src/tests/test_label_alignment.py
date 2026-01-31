"""
Test script for neuron label alignment feature.

Tests:
1. Global alignment (left, center, right)
2. Per-layer alignment overrides
3. Mixed alignments in same network
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer, VectorInput, VectorOutput
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig, LayerStyle

# Create output directory
# Create output directory at project root
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
output_dir = os.path.join(project_root, "test_outputs", "label_alignment")
os.makedirs(output_dir, exist_ok=True)

def test_global_alignment():
    """Test global alignment settings."""
    print("\nTesting global alignment settings...")
    
    nn = NeuralNetwork("Global Alignment Test")
    nn.add_layer(VectorInput(
        num_features=3,
        name="Input",
        neuron_labels=["A", "B", "C"],
        label_position="left"
    ))
    nn.add_layer(FullyConnectedLayer(3, activation="relu", name="Hidden"))
    nn.add_layer(FullyConnectedLayer(
        num_neurons=2,
        name="Output",
        neuron_labels=["X", "Y"],
        label_position="right"
    ))
    
    # Test left alignment
    config_left = PlotConfig(
        show_neuron_text_labels=True,
        neuron_text_label_alignment='left',
        background_color='white'
    )
    plot_network(
        nn,
        config=config_left,
        title="Global Left Alignment",
        save_path=os.path.join(output_dir, "01_global_left.png"),
        show=False
    )
    print("  ✓ Global left alignment")
    
    # Test center alignment
    config_center = PlotConfig(
        show_neuron_text_labels=True,
        neuron_text_label_alignment='center',
        background_color='white'
    )
    plot_network(
        nn,
        config=config_center,
        title="Global Center Alignment",
        save_path=os.path.join(output_dir, "02_global_center.png"),
        show=False
    )
    print("  ✓ Global center alignment")
    
    # Test right alignment
    config_right = PlotConfig(
        show_neuron_text_labels=True,
        neuron_text_label_alignment='right',
        background_color='white'
    )
    plot_network(
        nn,
        config=config_right,
        title="Global Right Alignment",
        save_path=os.path.join(output_dir, "03_global_right.png"),
        show=False
    )
    print("  ✓ Global right alignment")


def test_per_layer_alignment():
    """Test per-layer alignment overrides."""
    print("\nTesting per-layer alignment overrides...")
    
    nn = NeuralNetwork("Per-Layer Test")
    nn.add_layer(VectorInput(
        num_features=3,
        name="Input",
        neuron_labels=["Feature 1", "Feature 2", "Feature 3"],
        label_position="left"
    ))
    nn.add_layer(FullyConnectedLayer(3, activation="relu", name="Hidden"))
    nn.add_layer(VectorOutput(
        num_neurons=2,
        name="Output",
        neuron_labels=["Class A", "Class B"],
        label_position="right"
    ))
    
    config = PlotConfig(
        show_neuron_text_labels=True,
        neuron_text_label_alignment='center',  # Global default
        background_color='white',
        layer_styles={
            'Input': LayerStyle(
                neuron_fill_color='lightblue',
                neuron_text_label_alignment='left'  # Override to left
            ),
            'Output': LayerStyle(
                neuron_fill_color='lightcoral',
                neuron_text_label_alignment='right'  # Override to right
            )
        }
    )
    
    plot_network(
        nn,
        config=config,
        title="Per-Layer Alignment: Input=Left, Output=Right",
        save_path=os.path.join(output_dir, "04_per_layer_mixed.png"),
        show=False
    )
    print("  ✓ Per-layer alignment overrides")


def test_latex_alignment():
    """Test alignment with LaTeX labels."""
    print("\nTesting alignment with LaTeX labels...")
    
    nn = NeuralNetwork("LaTeX Alignment")
    nn.add_layer(VectorInput(
        num_features=3,
        name="Input",
        neuron_labels=[r"$x_1$", r"$x_2$", r"$x_3$"],
        label_position="left"
    ))
    nn.add_layer(FullyConnectedLayer(3, activation="relu", name="Hidden"))
    nn.add_layer(VectorOutput(
        num_neurons=2,
        name="Output",
        neuron_labels=[r"$\hat{y}_1$", r"$\hat{y}_2$"],
        label_position="right"
    ))
    
    # Test with per-layer alignment
    config = PlotConfig(
        show_neuron_text_labels=True,
        neuron_text_label_fontsize=12,
        background_color='white',
        layer_styles={
            'Input': LayerStyle(neuron_text_label_alignment='right'),
            'Output': LayerStyle(neuron_text_label_alignment='left')
        }
    )
    
    plot_network(
        nn,
        config=config,
        title="LaTeX with Per-Layer Alignment",
        save_path=os.path.join(output_dir, "05_latex_per_layer.png"),
        show=False
    )
    print("  ✓ LaTeX with per-layer alignment")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Neuron Label Alignment Test Suite")
    print("=" * 60)
    
    test_global_alignment()
    test_per_layer_alignment()
    test_latex_alignment()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("Check 'test_outputs/label_alignment' for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
