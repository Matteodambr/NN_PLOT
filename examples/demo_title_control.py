"""
Demo: Title Control

This script demonstrates the show_title option to control whether
the plot title is displayed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer, VectorInput
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig

# Create output directory
os.makedirs("outputs", exist_ok=True)

print("=" * 70)
print("TITLE CONTROL DEMONSTRATION")
print("=" * 70)

# Create a simple network
nn = NeuralNetwork("My Neural Network")
nn.add_layer(VectorInput(num_features=4, name="Input"))
nn.add_layer(FullyConnectedLayer(6, activation="relu", name="Hidden"))
nn.add_layer(FullyConnectedLayer(3, activation="softmax", name="Output"))

# ==============================================================================
# 1. With Title (Default Behavior)
# ==============================================================================
print("\n1. With Title (default behavior)")
print("-" * 70)

config_with_title = PlotConfig(show_title=True)  # This is the default

plot_network(
    nn,
    config=config_with_title,
    title="Custom Title: My Network Architecture",
    save_path="outputs/title_01_with_custom.png",
    show=False
)
print("✓ Created: outputs/title_01_with_custom.png")
print("  Title shown: 'Custom Title: My Network Architecture'")

# ==============================================================================
# 2. With Default Title
# ==============================================================================
print("\n2. With Default Title (when no title specified)")
print("-" * 70)

plot_network(
    nn,
    config=config_with_title,
    save_path="outputs/title_02_with_default.png",
    show=False
)
print("✓ Created: outputs/title_02_with_default.png")
print("  Title shown: 'Neural Network: My Neural Network' (default)")

# ==============================================================================
# 3. Without Title
# ==============================================================================
print("\n3. Without Title (clean plot)")
print("-" * 70)

config_no_title = PlotConfig(show_title=False)

plot_network(
    nn,
    config=config_no_title,
    save_path="outputs/title_03_no_title.png",
    show=False
)
print("✓ Created: outputs/title_03_no_title.png")
print("  No title shown - clean plot")

# ==============================================================================
# 4. No Title with Custom Styling
# ==============================================================================
print("\n4. No Title with Custom Styling")
print("-" * 70)

from src.NN_PLOTTING_UTILITIES import LayerStyle

layer_styles = {
    "Input": LayerStyle(
        neuron_fill_color="lightcoral",
        neuron_edge_color="darkred",
        connection_color="red",
        connection_alpha=0.4
    ),
    "Hidden": LayerStyle(
        neuron_fill_color="lightgreen",
        neuron_edge_color="darkgreen"
    ),
    "Output": LayerStyle(
        neuron_fill_color="lightskyblue",
        neuron_edge_color="navy"
    )
}

config_styled_no_title = PlotConfig(
    show_title=False,
    layer_styles=layer_styles
)

plot_network(
    nn,
    config=config_styled_no_title,
    save_path="outputs/title_04_styled_no_title.png",
    show=False
)
print("✓ Created: outputs/title_04_styled_no_title.png")
print("  Styled network without title")

# ==============================================================================
# 5. Comparison: With vs Without Title
# ==============================================================================
print("\n5. Minimal Network Comparison")
print("-" * 70)

config_minimal_with = PlotConfig(
    show_title=True,
    show_layer_names=False,
    show_neuron_labels=False
)

config_minimal_without = PlotConfig(
    show_title=False,
    show_layer_names=False,
    show_neuron_labels=False
)

plot_network(
    nn,
    config=config_minimal_with,
    title="Minimal View",
    save_path="outputs/title_05a_minimal_with.png",
    show=False
)
print("✓ Created: outputs/title_05a_minimal_with.png (with title)")

plot_network(
    nn,
    config=config_minimal_without,
    save_path="outputs/title_05b_minimal_without.png",
    show=False
)
print("✓ Created: outputs/title_05b_minimal_without.png (without title)")

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "=" * 70)
print("DEMONSTRATION COMPLETE!")
print("=" * 70)
print("\nCreated 6 example files in the 'outputs/' directory:")
print("\n  With Title:")
print("    title_01_with_custom.png - Custom title specified")
print("    title_02_with_default.png - Default title (network name)")
print("    title_05a_minimal_with.png - Minimal view with title")
print("\n  Without Title:")
print("    title_03_no_title.png - No title, clean plot")
print("    title_04_styled_no_title.png - Styled network without title")
print("    title_05b_minimal_without.png - Minimal view without title")
print("\n" + "=" * 70)
print("Usage:")
print("  config = PlotConfig(show_title=True)   # Show title (default)")
print("  config = PlotConfig(show_title=False)  # Hide title")
print("=" * 70)
