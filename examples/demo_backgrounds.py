"""
Demo: Background Color Options

This script demonstrates the different background color options available:
1. Transparent background (default)
2. White background
3. Custom color backgrounds (hex, rgb, named colors)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig

# Create output directory
os.makedirs("outputs", exist_ok=True)

print("=" * 70)
print("BACKGROUND COLOR DEMONSTRATION")
print("=" * 70)

# Create a simple network for demonstration
nn = NeuralNetwork("Background Demo")
nn.add_layer(FullyConnectedLayer(4, name="Input"))
nn.add_layer(FullyConnectedLayer(6, activation="relu", name="Hidden"))
nn.add_layer(FullyConnectedLayer(3, activation="softmax", name="Output"))

# ==============================================================================
# 1. Transparent Background (Default)
# ==============================================================================
print("\n1. Transparent Background (default)")
print("-" * 70)

config_transparent = PlotConfig(
    background_color='transparent'  # This is the default
)

plot_network(
    nn,
    config=config_transparent,
    title="Transparent Background (Default)",
    save_path="outputs/bg_01_transparent.png",
    show=False
)
print("✓ Created: outputs/bg_01_transparent.png")

# ==============================================================================
# 2. White Background
# ==============================================================================
print("\n2. White Background")
print("-" * 70)

config_white = PlotConfig(
    background_color='white'
)

plot_network(
    nn,
    config=config_white,
    title="White Background",
    save_path="outputs/bg_02_white.png",
    show=False
)
print("✓ Created: outputs/bg_02_white.png")

# ==============================================================================
# 3. Light Gray Background
# ==============================================================================
print("\n3. Light Gray Background")
print("-" * 70)

config_gray = PlotConfig(
    background_color='lightgray'
)

plot_network(
    nn,
    config=config_gray,
    title="Light Gray Background",
    save_path="outputs/bg_03_lightgray.png",
    show=False
)
print("✓ Created: outputs/bg_03_lightgray.png")

# ==============================================================================
# 4. Light Blue Background
# ==============================================================================
print("\n4. Light Blue Background")
print("-" * 70)

config_lightblue = PlotConfig(
    background_color='#E6F2FF'  # Hex color
)

plot_network(
    nn,
    config=config_lightblue,
    title="Light Blue Background",
    save_path="outputs/bg_04_lightblue.png",
    show=False
)
print("✓ Created: outputs/bg_04_lightblue.png")

# ==============================================================================
# 5. Cream Background
# ==============================================================================
print("\n5. Cream Background")
print("-" * 70)

config_cream = PlotConfig(
    background_color='#FFFEF0'
)

plot_network(
    nn,
    config=config_cream,
    title="Cream Background",
    save_path="outputs/bg_05_cream.png",
    show=False
)
print("✓ Created: outputs/bg_05_cream.png")

# ==============================================================================
# 6. Dark Mode (Dark Background)
# ==============================================================================
print("\n6. Dark Mode Background")
print("-" * 70)

config_dark = PlotConfig(
    background_color='#2B2B2B',
    neuron_color='lightblue',
    neuron_edge_color='white',
    connection_color='lightgray',
    connection_alpha=0.5
)

plot_network(
    nn,
    config=config_dark,
    title="Dark Mode Background",
    save_path="outputs/bg_06_dark.png",
    show=False
)
print("✓ Created: outputs/bg_06_dark.png")

# ==============================================================================
# 7. SVG with Transparent Background
# ==============================================================================
print("\n7. SVG with Transparent Background")
print("-" * 70)

plot_network(
    nn,
    config=config_transparent,
    title="Transparent SVG Format",
    save_path="outputs/bg_07_transparent.svg",
    show=False,
    format="svg"
)
print("✓ Created: outputs/bg_07_transparent.svg")

# ==============================================================================
# 8. PDF with White Background
# ==============================================================================
print("\n8. PDF with White Background")
print("-" * 70)

plot_network(
    nn,
    config=config_white,
    title="White Background PDF",
    save_path="outputs/bg_08_white.pdf",
    show=False,
    format="pdf"
)
print("✓ Created: outputs/bg_08_white.pdf")

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "=" * 70)
print("DEMONSTRATION COMPLETE!")
print("=" * 70)
print("\nCreated 8 example files in the 'outputs/' directory:")
print("\n  PNG Files:")
print("    bg_01_transparent.png - Transparent background (default)")
print("    bg_02_white.png - White background")
print("    bg_03_lightgray.png - Light gray background")
print("    bg_04_lightblue.png - Light blue background (#E6F2FF)")
print("    bg_05_cream.png - Cream background (#FFFEF0)")
print("    bg_06_dark.png - Dark mode (#2B2B2B)")
print("\n  Vector Formats:")
print("    bg_07_transparent.svg - SVG with transparent background")
print("    bg_08_white.pdf - PDF with white background")
print("\n" + "=" * 70)
print("Background Color Options:")
print("  - 'transparent' (default) - No background, transparent PNG/SVG")
print("  - 'white' - White background")
print("  - Any named color: 'lightgray', 'lightblue', 'cream', etc.")
print("  - Hex colors: '#E6F2FF', '#2B2B2B', etc.")
print("  - RGB tuples: (0.9, 0.9, 0.9) for light gray")
print("=" * 70)
