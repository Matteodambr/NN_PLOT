"""
COMPREHENSIVE SHOWCASE - All NN_PLOT Features

This script demonstrates ALL features of the NN_PLOT library in one place:
1. Basic network creation and plotting
2. Layer-specific styling (colors, edges, connections)
3. Custom neuron labels (text and LaTeX math)
4. Neuron numbering (normal and reversed)
5. Layer collapsing for large networks
6. Different export formats (PNG, SVG, PDF)
7. DPI control for image quality
8. Background colors (transparent, white, custom colors)
9. Font customization
10. Show/hide various elements
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer, VectorInput
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig, LayerStyle

# Create output directory
os.makedirs("outputs", exist_ok=True)

print("=" * 70)
print("NN_PLOT - COMPREHENSIVE FEATURE DEMONSTRATION")
print("=" * 70)

# ==============================================================================
# EXAMPLE 1: Basic Network
# ==============================================================================
print("\n1. Basic Network")
print("-" * 70)

nn_basic = NeuralNetwork("Basic Neural Network")
nn_basic.add_layer(VectorInput(num_features=4, name="Input"))
nn_basic.add_layer(FullyConnectedLayer(6, activation="relu", name="Hidden"))
nn_basic.add_layer(FullyConnectedLayer(3, activation="softmax", name="Output"))

plot_network(
    nn_basic,
    title="Example 1: Basic Network",
    save_path="outputs/01_basic_network.png",
    show=False
)
print("✓ Created: outputs/01_basic_network.png")

# ==============================================================================
# EXAMPLE 2: Custom Layer Styling
# ==============================================================================
print("\n2. Custom Layer Styling")
print("-" * 70)

nn_styled = NeuralNetwork("Styled Network")
nn_styled.add_layer(VectorInput(num_features=4, name="Input"))
nn_styled.add_layer(FullyConnectedLayer(6, activation="relu", name="Hidden"))
nn_styled.add_layer(FullyConnectedLayer(3, activation="softmax", name="Output"))

# Define custom styles for each layer
layer_styles = {
    "Input": LayerStyle(
        neuron_fill_color="lightcoral",
        neuron_edge_color="darkred",
        neuron_edge_width=2.5,
        connection_color="red",
        connection_alpha=0.4
    ),
    "Hidden": LayerStyle(
        neuron_fill_color="lightgreen",
        neuron_edge_color="darkgreen",
        neuron_edge_width=2.0,
        connection_color="green"
    ),
    "Output": LayerStyle(
        neuron_fill_color="lightskyblue",
        neuron_edge_color="navy",
        neuron_edge_width=3.0
    )
}

config_styled = PlotConfig(layer_styles=layer_styles)
plot_network(
    nn_styled,
    config=config_styled,
    title="Example 2: Custom Layer Styling",
    save_path="outputs/02_styled_network.png",
    show=False
)
print("✓ Created: outputs/02_styled_network.png")

# ==============================================================================
# EXAMPLE 3: Custom Neuron Labels with Plain Text
# ==============================================================================
print("\n3. Custom Neuron Labels (Plain Text)")
print("-" * 70)

nn_labels = NeuralNetwork("Labeled Network")
nn_labels.add_layer(VectorInput(
    num_features=5,
    name="Input Features",
    neuron_labels=[
        "Age",
        "Income",
        "Credit Score",
        "Employment Years",
        "Loan Amount"
    ],
    label_position="left"
))
nn_labels.add_layer(FullyConnectedLayer(8, activation="relu", name="Hidden Layer"))
nn_labels.add_layer(FullyConnectedLayer(
    num_neurons=3,
    name="Risk Category",
    neuron_labels=["Low Risk", "Medium Risk", "High Risk"],
    label_position="right"
))

config_labels = PlotConfig(
    show_neuron_text_labels=True,
    neuron_text_label_fontsize=10
)
plot_network(
    nn_labels,
    config=config_labels,
    title="Example 3: Plain Text Neuron Labels",
    save_path="outputs/03_text_labels.png",
    show=False
)
print("✓ Created: outputs/03_text_labels.png")

# ==============================================================================
# EXAMPLE 4: LaTeX Mathematical Notation
# ==============================================================================
print("\n4. LaTeX Mathematical Notation")
print("-" * 70)

nn_latex = NeuralNetwork("LaTeX Math Network")
nn_latex.add_layer(VectorInput(
    num_features=3,
    name="Input",
    neuron_labels=[r"$x_1$", r"$x_2$", r"$x_3$"],
    label_position="left"
))
nn_latex.add_layer(FullyConnectedLayer(
    num_neurons=5,
    activation="tanh",
    name="Hidden",
    neuron_labels=[r"$h_1$", r"$h_2$", r"$h_3$", r"$h_4$", r"$h_5$"],
    label_position="left"
))
nn_latex.add_layer(FullyConnectedLayer(
    num_neurons=2,
    name="Output",
    neuron_labels=[r"$\hat{y}_1$", r"$\hat{y}_2$"],
    label_position="right"
))

config_latex = PlotConfig(
    show_neuron_text_labels=True,
    neuron_text_label_fontsize=12
)
plot_network(
    nn_latex,
    config=config_latex,
    title="Example 4: LaTeX Mathematical Notation",
    save_path="outputs/04_latex_math.png",
    show=False
)
print("✓ Created: outputs/04_latex_math.png")

# ==============================================================================
# EXAMPLE 5: Bold Math and Complex LaTeX
# ==============================================================================
print("\n5. Bold Math and Complex LaTeX")
print("-" * 70)

nn_bold = NeuralNetwork("Bold Math Network")
nn_bold.add_layer(VectorInput(
    num_features=4,
    name="Input",
    neuron_labels=[
        r"$\mathbf{x}_1$",      # Bold
        r"$\boldsymbol{\alpha}$",  # Bold Greek
        r"$\frac{a}{b}$",       # Fraction
        r"$\sum_{i=1}^n x_i$"   # Summation
    ],
    label_position="left"
))
nn_bold.add_layer(FullyConnectedLayer(4, activation="relu", name="Hidden"))
nn_bold.add_layer(FullyConnectedLayer(
    num_neurons=2,
    name="Output",
    neuron_labels=[r"$\boldsymbol{\theta}$", r"$\boldsymbol{\phi}$"],
    label_position="right"
))

plot_network(
    nn_bold,
    config=config_latex,
    title="Example 5: Bold Math and Complex LaTeX",
    save_path="outputs/05_bold_latex.png",
    show=False
)
print("✓ Created: outputs/05_bold_latex.png")

# ==============================================================================
# EXAMPLE 6: Neuron Numbering (Normal and Reversed)
# ==============================================================================
print("\n6. Neuron Numbering")
print("-" * 70)

nn_numbering = NeuralNetwork("Numbered Network")
nn_numbering.add_layer(VectorInput(num_features=6, name="Input"))
nn_numbering.add_layer(FullyConnectedLayer(4, activation="relu", name="Hidden"))
nn_numbering.add_layer(FullyConnectedLayer(3, activation="softmax", name="Output"))

# Normal numbering (0 at top)
config_normal = PlotConfig(show_neuron_labels=True, neuron_numbering_reversed=False)
plot_network(
    nn_numbering,
    config=config_normal,
    title="Example 6a: Normal Numbering (0 at top)",
    save_path="outputs/06a_numbering_normal.png",
    show=False
)
print("✓ Created: outputs/06a_numbering_normal.png")

# Reversed numbering (N-1 at top)
config_reversed = PlotConfig(show_neuron_labels=True, neuron_numbering_reversed=True)
plot_network(
    nn_numbering,
    config=config_reversed,
    title="Example 6b: Reversed Numbering (N-1 at top)",
    save_path="outputs/06b_numbering_reversed.png",
    show=False
)
print("✓ Created: outputs/06b_numbering_reversed.png")

# ==============================================================================
# EXAMPLE 7: Layer Collapsing for Large Networks
# ==============================================================================
print("\n7. Layer Collapsing")
print("-" * 70)

nn_large = NeuralNetwork("Large Network with Collapsing")
nn_large.add_layer(VectorInput(
    num_features=25,
    name="Large Input",
    neuron_labels=[f"Feature {i+1}" for i in range(25)],
    label_position="left"
))
nn_large.add_layer(FullyConnectedLayer(20, activation="relu", name="Hidden 1"))
nn_large.add_layer(FullyConnectedLayer(15, activation="relu", name="Hidden 2"))
nn_large.add_layer(FullyConnectedLayer(
    num_neurons=5,
    activation="softmax",
    name="Output",
    neuron_labels=[f"Class {i}" for i in range(5)],
    label_position="right"
))

config_collapsed = PlotConfig(
    show_neuron_text_labels=True,
    max_neurons_per_layer=12,
    collapse_neurons_start=5,
    collapse_neurons_end=5
)
plot_network(
    nn_large,
    config=config_collapsed,
    title="Example 7: Collapsed Large Layers",
    save_path="outputs/07_collapsed_network.png",
    show=False
)
print("✓ Created: outputs/07_collapsed_network.png")

# ==============================================================================
# EXAMPLE 8: Different Export Formats and DPI
# ==============================================================================
print("\n8. Export Formats and DPI Control")
print("-" * 70)

nn_export = NeuralNetwork("Export Demo")
nn_export.add_layer(VectorInput(num_features=3, name="Input"))
nn_export.add_layer(FullyConnectedLayer(4, activation="relu", name="Hidden"))
nn_export.add_layer(FullyConnectedLayer(2, activation="sigmoid", name="Output"))

# Low DPI PNG (quick preview)
plot_network(
    nn_export,
    title="Low DPI (72)",
    save_path="outputs/08a_low_dpi.png",
    show=False,
    dpi=72
)
print("✓ Created: outputs/08a_low_dpi.png (72 DPI)")

# High DPI PNG (publication quality)
plot_network(
    nn_export,
    title="High DPI (600)",
    save_path="outputs/08b_high_dpi.png",
    show=False,
    dpi=600
)
print("✓ Created: outputs/08b_high_dpi.png (600 DPI)")

# SVG format (scalable vector)
plot_network(
    nn_export,
    title="SVG Format (Scalable)",
    save_path="outputs/08c_vector.svg",
    show=False,
    format="svg"
)
print("✓ Created: outputs/08c_vector.svg (SVG)")

# PDF format
plot_network(
    nn_export,
    title="PDF Format",
    save_path="outputs/08d_document.pdf",
    show=False,
    format="pdf"
)
print("✓ Created: outputs/08d_document.pdf (PDF)")

# ==============================================================================
# EXAMPLE 9: Background Colors
# ==============================================================================
print("\n9. Background Colors")
print("-" * 70)

nn_bg = NeuralNetwork("Background Demo")
nn_bg.add_layer(VectorInput(num_features=3, name="Input"))
nn_bg.add_layer(FullyConnectedLayer(4, activation="relu", name="Hidden"))
nn_bg.add_layer(FullyConnectedLayer(2, activation="sigmoid", name="Output"))

# Transparent background (default)
config_transparent = PlotConfig(background_color='transparent')
plot_network(
    nn_bg,
    config=config_transparent,
    title="Transparent Background (Default)",
    save_path="outputs/09a_transparent.png",
    show=False
)
print("✓ Created: outputs/09a_transparent.png (transparent - default)")

# White background
config_white = PlotConfig(background_color='white')
plot_network(
    nn_bg,
    config=config_white,
    title="White Background",
    save_path="outputs/09b_white.png",
    show=False
)
print("✓ Created: outputs/09b_white.png (white)")

# Custom color background
config_custom = PlotConfig(background_color='#E6F2FF')
plot_network(
    nn_bg,
    config=config_custom,
    title="Custom Color Background",
    save_path="outputs/09c_custom_color.png",
    show=False
)
print("✓ Created: outputs/09c_custom_color.png (light blue)")

# ==============================================================================
# EXAMPLE 10: Font Customization
# ==============================================================================
print("\n10. Font Customization")
print("-" * 70)

nn_font = NeuralNetwork("Font Demo")
nn_font.add_layer(VectorInput(
    num_features=3,
    name="Input",
    neuron_labels=[r"$x_1$", r"$x_2$", r"$x_3$"],
    label_position="left"
))
nn_font.add_layer(FullyConnectedLayer(4, activation="relu", name="Hidden"))
nn_font.add_layer(FullyConnectedLayer(
    num_neurons=2,
    name="Output",
    neuron_labels=["Class A", "Class B"],
    label_position="right"
))

# Times New Roman (serif)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

config_font = PlotConfig(show_neuron_text_labels=True)
plot_network(
    nn_font,
    config=config_font,
    title="Example 10a: Times New Roman Font",
    save_path="outputs/10a_times_font.png",
    show=False
)
print("✓ Created: outputs/10a_times_font.png (Times New Roman)")

# Reset and use Arial
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

plot_network(
    nn_font,
    config=config_font,
    title="Example 10b: Arial Font",
    save_path="outputs/10b_arial_font.png",
    show=False
)
print("✓ Created: outputs/10b_arial_font.png (Arial)")

# Reset to default
plt.rcParams.update(plt.rcParamsDefault)

# ==============================================================================
# EXAMPLE 11: Show/Hide Elements
# ==============================================================================
print("\n11. Show/Hide Various Elements")
print("-" * 70)

nn_visibility = NeuralNetwork("Visibility Demo")
nn_visibility.add_layer(VectorInput(
    num_features=4,
    name="Input Layer",
    neuron_labels=["A", "B", "C", "D"],
    label_position="left"
))
nn_visibility.add_layer(FullyConnectedLayer(5, activation="relu", name="Hidden Layer"))
nn_visibility.add_layer(FullyConnectedLayer(
    num_neurons=2,
    name="Output Layer",
    neuron_labels=["Y1", "Y2"],
    label_position="right"
))

# All elements visible
config_all = PlotConfig(
    show_layer_names=True,
    show_neuron_labels=True,
    show_neuron_text_labels=True
)
plot_network(
    nn_visibility,
    config=config_all,
    title="Example 11a: All Elements Visible",
    save_path="outputs/11a_all_visible.png",
    show=False
)
print("✓ Created: outputs/11a_all_visible.png (all visible)")

# Only custom labels
config_labels_only = PlotConfig(
    show_layer_names=False,
    show_neuron_labels=False,
    show_neuron_text_labels=True
)
plot_network(
    nn_visibility,
    config=config_labels_only,
    title="Example 11b: Custom Labels Only",
    save_path="outputs/11b_labels_only.png",
    show=False
)
print("✓ Created: outputs/11b_labels_only.png (custom labels only)")

# Minimal view
config_minimal = PlotConfig(
    show_layer_names=True,
    show_neuron_labels=False,
    show_neuron_text_labels=False
)
plot_network(
    nn_visibility,
    config=config_minimal,
    title="Example 11c: Minimal View",
    save_path="outputs/11c_minimal.png",
    show=False
)
print("✓ Created: outputs/11c_minimal.png (minimal)")

# ==============================================================================
# EXAMPLE 12: Combined Features (Realistic Use Case)
# ==============================================================================
print("\n12. Combined Features - Realistic Example")
print("-" * 70)

nn_realistic = NeuralNetwork("Customer Churn Prediction Model")

# Input with styling and labels
nn_realistic.add_layer(VectorInput(
    num_features=8,
    name="Customer Features",
    neuron_labels=[
        "Account Age",
        "Monthly Charges",
        "Total Charges",
        "Contract Type",
        "Payment Method",
        "Tech Support",
        "Online Security",
        "Device Protection"
    ],
    label_position="left"
))

nn_realistic.add_layer(FullyConnectedLayer(
    num_neurons=12,
    activation="relu",
    name="Feature Processing"
))

nn_realistic.add_layer(FullyConnectedLayer(
    num_neurons=8,
    activation="relu",
    name="Pattern Recognition"
))

nn_realistic.add_layer(FullyConnectedLayer(
    num_neurons=4,
    activation="relu",
    name="Decision Layer"
))

nn_realistic.add_layer(FullyConnectedLayer(
    num_neurons=2,
    activation="softmax",
    name="Churn Prediction",
    neuron_labels=["Will Stay", "Will Churn"],
    label_position="right"
))

# Custom styling
realistic_styles = {
    "Customer Features": LayerStyle(
        neuron_fill_color="lightblue",
        neuron_edge_color="darkblue",
        neuron_edge_width=2.0,
        connection_color="blue",
        connection_alpha=0.3
    ),
    "Churn Prediction": LayerStyle(
        neuron_fill_color="lightcoral",
        neuron_edge_color="darkred",
        neuron_edge_width=2.5
    )
}

config_realistic = PlotConfig(
    show_neuron_text_labels=True,
    neuron_text_label_fontsize=9,
    show_layer_names=True,
    layer_styles=realistic_styles,
    figsize=(16, 10)
)

plot_network(
    nn_realistic,
    config=config_realistic,
    title="Example 12: Complete Customer Churn Prediction Model",
    save_path="outputs/12_realistic_combined.png",
    show=False,
    dpi=300
)
print("✓ Created: outputs/12_realistic_combined.png (combined features)")

# Also save as SVG for presentations
plot_network(
    nn_realistic,
    config=config_realistic,
    title="Customer Churn Prediction Model",
    save_path="outputs/12_realistic_combined.svg",
    show=False,
    format="svg"
)
print("✓ Created: outputs/12_realistic_combined.svg (SVG for presentations)")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 70)
print("DEMONSTRATION COMPLETE!")
print("=" * 70)
print("\nCreated 20 example files in the 'outputs/' directory:")
print("\n  Basic Features:")
print("    01_basic_network.png")
print("    02_styled_network.png")
print("\n  Custom Labels:")
print("    03_text_labels.png")
print("    04_latex_math.png")
print("    05_bold_latex.png")
print("\n  Neuron Numbering:")
print("    06a_numbering_normal.png")
print("    06b_numbering_reversed.png")
print("\n  Large Networks:")
print("    07_collapsed_network.png")
print("\n  Export Formats:")
print("    08a_low_dpi.png (72 DPI)")
print("    08b_high_dpi.png (600 DPI)")
print("    08c_vector.svg (SVG)")
print("    08d_document.pdf (PDF)")
print("\n  Background Colors:")
print("    09a_transparent.png (transparent - default)")
print("    09b_white.png (white)")
print("    09c_custom_color.png (light blue)")
print("\n  Fonts:")
print("    10a_times_font.png")
print("    10b_arial_font.png")
print("\n  Visibility:")
print("    11a_all_visible.png")
print("    11b_labels_only.png")
print("    11c_minimal.png")
print("\n  Complete Example:")
print("    12_realistic_combined.png")
print("    12_realistic_combined.svg")
print("\n" + "=" * 70)
print("All features demonstrated successfully!")
print("=" * 70)
