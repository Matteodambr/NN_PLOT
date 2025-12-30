"""
Comprehensive Encoder-Decoder Showcase
=======================================
This example demonstrates ALL labeling and customization features:
- Neuron text labels (LaTeX math)
- Layer names with curly braces
- Layer variable names
- Layer grouping with custom brackets
- Custom colors and styling
- Alignment options

Perfect for README/documentation images!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib
matplotlib.use('Agg')

from NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer
from NN_PLOTTING_UTILITIES import plot_network, PlotConfig, LayerGroup, LayerStyle

os.makedirs("showcase_outputs", exist_ok=True)

print("=" * 70)
print("Creating Comprehensive Encoder-Decoder Showcase")
print("=" * 70)

# Create encoder-decoder autoencoder network
nn = NeuralNetwork("Autoencoder for Dimensionality Reduction")

# Input layer
input_id = nn.add_layer(FullyConnectedLayer(
    num_neurons=256,
    name="Input Layer",
    neuron_labels=[f"$x_{{{i}}}$" for i in range(256, 0, -1)],  # Full array of labels
    label_position="left"
))

# Encoder layers
enc1_id = nn.add_layer(FullyConnectedLayer(
    num_neurons=128,
    activation="relu",
    name="Encoder 1"
), parent_ids=[input_id])

enc2_id = nn.add_layer(FullyConnectedLayer(
    num_neurons=4,
    activation="relu",
    name="Encoder 2"
), parent_ids=[enc1_id])

# Latent space (bottleneck)
latent_id = nn.add_layer(FullyConnectedLayer(
    num_neurons=2,
    activation="linear",
    name="Latent Space",
    neuron_labels=[r"$z_2$", r"$z_1$"],
    label_position="right"
), parent_ids=[enc2_id])

# Decoder layers
dec1_id = nn.add_layer(FullyConnectedLayer(
    num_neurons=4,
    activation="relu",
    name="Decoder 1"
), parent_ids=[latent_id])

dec2_id = nn.add_layer(FullyConnectedLayer(
    num_neurons=128,
    activation="relu",
    name="Decoder 2"
), parent_ids=[dec1_id])

# Output layer
output_id = nn.add_layer(FullyConnectedLayer(
    num_neurons=256,
    activation="sigmoid",
    name="Output Layer",
    neuron_labels=[r"$\hat{x}_{" + str(i) + "}$" for i in range(256, 0, -1)],  # Full array of labels
    label_position="right"
), parent_ids=[dec2_id])

# Create comprehensive configuration
config = PlotConfig(
    # Figure settings
    figsize=(16, 10),
    background_color='white',
    
    # Neuron appearance
    neuron_radius=0.35,
    neuron_color='lightblue',
    neuron_edge_color='navy',
    neuron_edge_width=2.0,
    
    # Neuron text labels
    show_neuron_text_labels=True,
    neuron_text_label_fontsize=18,
    neuron_text_label_offset=0.85,
    
    # Connections
    connection_alpha=0.65,
    connection_color='gray',
    connection_linewidth=2.0,
    
    # Neuron collapse settings (applied when num_neurons > max_neurons_per_layer)
    max_neurons_per_layer=10,  # Will collapse 256, 128, and 4-neuron layers
    collapse_neurons_start=4,
    collapse_neurons_end=4,
    
    # Layer names (below each layer with curly braces)
    show_layer_names=True,
    layer_name_fontsize=12,
    layer_names_line_styles=[],  # No brackets for individual layers
    layer_names_show_type=False,
    layer_names_show_dim=True,
    layer_names_show_activation=True,
    layer_names_align_bottom=True,
    layer_names_bottom_offset=1.8,
    
    # Layer variable names (high-level descriptions)
    show_layer_variable_names=True,
    layer_variable_names={
        'Input Layer': 'Input\nFeatures',
        'Latent Space': 'Latent\nRepresentation',
        'Output Layer': 'Reconstructed\nOutput'
    },
    layer_variable_names_fontsize=13,
    layer_variable_names_position='side',
    layer_variable_names_multialignment='center',
    layer_variable_names_offset=1.2,
    
    # Title
    title_fontsize=18,
    title_offset=12,
    
    # Layer-specific styling
    layer_styles={
        'Input Layer': LayerStyle(
            neuron_fill_color='#E8F4F8',
            neuron_edge_color='#1976D2',
            neuron_edge_width=2.5,
            box_around_layer=True,
            box_fill_color='#E3F2FD',
            box_edge_color='#1976D2',
            box_edge_width=2.5,
            box_padding=0.6,
            box_corner_radius=0.4,
            box_include_neuron_labels=True,
            max_neurons_to_plot=10,
            collapse_neurons_start=4,
            collapse_neurons_end=4,
            layer_name_bold=True,
            variable_name_color='#E3F2FD'
        ),
        'Encoder 1': LayerStyle(
            neuron_fill_color='#E8F4F8',
            neuron_edge_color='#1976D2',
            max_neurons_to_plot=10,
            collapse_neurons_start=3,
            collapse_neurons_end=3,
            layer_name_bold=True
        ),
        'Encoder 2': LayerStyle(
            neuron_fill_color='#E8F4F8',
            neuron_edge_color='#1976D2',
            max_neurons_to_plot=10,
            collapse_neurons_start=2,
            collapse_neurons_end=2,
            layer_name_bold=True
        ),
        'Latent Space': LayerStyle(
            neuron_fill_color='#FFF3E0',
            neuron_edge_color='#F57C00',
            neuron_edge_width=3.0,
            layer_name_bold=True,
            variable_name_color='#FFF3E0'
        ),
        'Decoder 1': LayerStyle(
            neuron_fill_color='#FFEBEE',
            neuron_edge_color='#D32F2F',
            max_neurons_to_plot=10,
            collapse_neurons_start=2,
            collapse_neurons_end=2,
            layer_name_bold=True
        ),
        'Decoder 2': LayerStyle(
            neuron_fill_color='#FFEBEE',
            neuron_edge_color='#D32F2F',
            max_neurons_to_plot=10,
            collapse_neurons_start=3,
            collapse_neurons_end=3,
            layer_name_bold=True
        ),
        'Output Layer': LayerStyle(
            neuron_fill_color='#FCE4EC',
            neuron_edge_color='#C2185B',
            neuron_edge_width=2.5,
            box_around_layer=True,
            box_fill_color='#F8BBD0',
            box_edge_color='#C2185B',
            box_edge_width=2.5,
            box_padding=0.6,
            box_corner_radius=0.4,
            box_include_neuron_labels=True,
            max_neurons_to_plot=10,
            collapse_neurons_start=4,
            collapse_neurons_end=4,
            layer_name_bold=True,
            variable_name_color='#F8BBD0'
        )
    },
    
    # Layer grouping with brackets
    layer_groups=[
        LayerGroup(
            layer_ids=['Input Layer', 'Encoder 1', 'Encoder 2'],
            label='Encoder Network',
            bracket_style='curly',
            bracket_color='#1976D2',
            bracket_linewidth=2.5,
            bracket_height=0.4,
            label_fontsize=16,
            label_color='#1976D2',
            y_offset=-2.0
        ),
        LayerGroup(
            layer_ids=['Decoder 1', 'Decoder 2', 'Output Layer'],
            label='Decoder Network',
            bracket_style='curly',
            bracket_color='#D32F2F',
            bracket_linewidth=2.5,
            bracket_height=0.4,
            label_fontsize=16,
            label_color='#D32F2F',
            y_offset=-2.0
        )
    ],
    
    # Font
    font_family='Times New Roman'
)

# Generate the plot
print("\nGenerating comprehensive showcase plot...")

plot_network(
    nn,
    config=config,
    title="Autoencoder Architecture: Complete Feature Showcase",
    save_path="showcase_outputs/encoder_decoder_comprehensive.png",
    show=False,
    dpi=300
)

print("✓ Saved: showcase_outputs/encoder_decoder_comprehensive.png")

print("\n" + "=" * 70)
print("Showcase Complete!")
print("=" * 70)
print("\nThis example demonstrates:")
print("  ✓ Neuron text labels with LaTeX math (left/right aligned)")
print("  ✓ Layer names with customizable curly braces")
print("  ✓ Layer variable names (high-level descriptions)")
print("  ✓ Layer grouping with different bracket styles")
print("  ✓ Custom colors per layer")
print("  ✓ Custom bracket colors, widths, and heights")
print("  ✓ All three label types displayed together")
print("\nPerfect for your repository README!")
print("=" * 70)
