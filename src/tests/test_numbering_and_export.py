"""
Test script demonstrating neuron numbering direction and export format options.

This script creates various network plots showcasing:
1. Normal neuron numbering (0 at top, N-1 at bottom)
2. Reversed neuron numbering (N-1 at top, 0 at bottom)
3. Different DPI values for quality control
4. SVG format export for scalable vector graphics
5. PNG export with custom DPI settings
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.NN_DEFINITION_UTILITIES import (
    NeuralNetwork,
    FullyConnectedLayer,
    VectorInput,
    VectorOutput
)

from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig

# Create output directory
# Create output directory relative to this script
# Create output directory at project root
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
output_dir = os.path.join(project_root, "test_outputs", "numbering_export")
os.makedirs(output_dir, exist_ok=True)

def test_numbering_directions():
    """Test normal vs reversed neuron numbering."""
    print("Testing neuron numbering directions...")
    
    # Create a simple network
    nn = NeuralNetwork("Numbering Test")
    nn.add_layer(VectorInput(num_features=8, name="Input"))
    nn.add_layer(FullyConnectedLayer(6, activation="relu", name="Hidden"))
    nn.add_layer(FullyConnectedLayer(3, activation="softmax", name="Output"))
    
    # Normal numbering (0 at top)
    config_normal = PlotConfig(
        show_neuron_labels=True,
        neuron_numbering_reversed=False
    )
    plot_network(
        nn,
        config=config_normal,
        title="Normal Numbering (0 at top)",
        save_path=os.path.join(output_dir, r"01_normal_numbering.png"),
        show=False
    )
    print("  ✓ Normal numbering (0 at top)")
    
    # Reversed numbering (N-1 at top)
    config_reversed = PlotConfig(
        show_neuron_labels=True,
        neuron_numbering_reversed=True
    )
    plot_network(
        nn,
        config=config_reversed,
        title="Reversed Numbering (N-1 at top)",
        save_path=os.path.join(output_dir, r"02_reversed_numbering.png"),
        show=False
    )
    print("  ✓ Reversed numbering (N-1 at top)")

def test_dpi_variations():
    """Test different DPI settings for image quality."""
    print("\nTesting DPI variations...")
    
    nn = NeuralNetwork("DPI Test")
    nn.add_layer(VectorInput(num_features=5, name="Input"))
    nn.add_layer(FullyConnectedLayer(3, activation="relu", name="Hidden"))
    nn.add_layer(FullyConnectedLayer(2, activation="sigmoid", name="Output"))
    
    config = PlotConfig(show_neuron_labels=True)
    
    dpi_values = [72, 150, 300, 600]
    for dpi in dpi_values:
        plot_network(
            nn,
            config=config,
            title=f"Network at {dpi} DPI",
            save_path=f"test_outputs/numbering_export/03_network_{dpi}dpi.png",
            show=False,
            dpi=dpi
        )
        print(f"  ✓ Saved at {dpi} DPI")

def test_export_formats():
    """Test different export formats (PNG, SVG, PDF)."""
    print("\nTesting export formats...")
    
    nn = NeuralNetwork("Format Test")
    nn.add_layer(VectorInput(num_features=6, name="Input"))
    nn.add_layer(FullyConnectedLayer(4, activation="relu", name="Hidden"))
    nn.add_layer(FullyConnectedLayer(2, activation="softmax", name="Output"))
    
    config = PlotConfig(
        show_neuron_labels=True,
        neuron_numbering_reversed=True
    )
    
    # PNG format (default)
    plot_network(
        nn,
        config=config,
        title="PNG Export",
        save_path=os.path.join(output_dir, r"04_network.png"),
        show=False,
        dpi=300
    )
    print("  ✓ PNG export")
    
    # SVG format (scalable vector graphics)
    plot_network(
        nn,
        config=config,
        title="SVG Export",
        save_path=os.path.join(output_dir, r"05_network.svg"),
        show=False,
        format="svg"
    )
    print("  ✓ SVG export")
    
    # PDF format
    plot_network(
        nn,
        config=config,
        title="PDF Export",
        save_path=os.path.join(output_dir, r"06_network.pdf"),
        show=False,
        format="pdf"
    )
    print("  ✓ PDF export")
    
    # Auto-detect format from extension
    plot_network(
        nn,
        config=config,
        title="Auto-detect Format",
        save_path=os.path.join(output_dir, r"07_network_auto.svg"),
        show=False
    )
    print("  ✓ Auto-detected format from extension (.svg)")

def test_combined_features():
    """Test combination of all new features."""
    print("\nTesting combined features...")
    
    nn = NeuralNetwork("Combined Features")
    nn.add_layer(VectorInput(num_features=10, name="Input"))
    nn.add_layer(FullyConnectedLayer(8, activation="relu", name="Hidden 1"))
    nn.add_layer(FullyConnectedLayer(6, activation="relu", name="Hidden 2"))
    nn.add_layer(VectorOutput(3, activation="softmax", name="Output"))
    
    # Collapsed layers with reversed numbering
    config_collapsed = PlotConfig(
        show_neuron_labels=True,
        neuron_numbering_reversed=True,
        collapse_neurons_start=3,
        collapse_neurons_end=3
    )
    
    plot_network(
        nn,
        config=config_collapsed,
        title="Collapsed + Reversed + High DPI",
        save_path=os.path.join(output_dir, r"08_combined_features.png"),
        show=False,
        dpi=600
    )
    print("  ✓ Collapsed layers + reversed numbering + 600 DPI")
    
    # Same network in SVG
    plot_network(
        nn,
        config=config_collapsed,
        title="Collapsed + Reversed (SVG)",
        save_path=os.path.join(output_dir, r"09_combined_features.svg"),
        show=False,
        format="svg"
    )
    print("  ✓ Same network exported as SVG")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Neural Network Numbering and Export Options Test Suite")
    print("=" * 60)
    
    test_numbering_directions()
    test_dpi_variations()
    test_export_formats()
    test_combined_features()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("Check the 'test_outputs/numbering_export' directory for results.")
    print("=" * 60)

if __name__ == "__main__":
    main()
