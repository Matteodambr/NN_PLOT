"""
CEAS 2025 Conference Paper - Network Architecture Figures

This script generates neural network architecture diagrams for the CEAS 2025 paper.

Paper: [Add your paper title here]
Authors: [Add authors]
Conference: CEAS 2025 (Council of European Aerospace Societies)

Figures generated:
- figure1_network_architecture.pdf: Main network architecture
- [Add more figure descriptions as needed]
"""

import sys
import os

# Add the NN_PLOT src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure matplotlib backend for non-interactive (file-only) mode
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive, file-only output

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig, LayerStyle

# ==============================================================================
# CONFIGURATION SECTION - Adjust these parameters for all figures
# ==============================================================================

# Figure layout and size
FIGSIZE = (14, 10)  # (width, height) in inches - larger for branching networks
BACKGROUND_COLOR = 'white'  # 'white', 'transparent', or any matplotlib color

# Display options
SHOW_TITLE = False  # Set to True to show figure titles
SHOW_LAYER_NAMES = False  # Set to True to show layer names below layers
SHOW_NEURON_LABELS = False  # Set to True to show neuron numbering (0, 1, 2, ...)
NEURON_NUMBERING_REVERSED = True  # True: labels specified top-to-bottom, False: bottom-to-top

# Neuron text labels (custom labels like $x_1$, $a_{1,1}$, etc.)
SHOW_NEURON_TEXT_LABELS = True  # Enable/disable custom neuron labels
NEURON_TEXT_LABEL_FONTSIZE = 24  # Font size for neuron labels
NEURON_TEXT_LABEL_OFFSET = 1.6  # Distance from neuron center (in plot units)

# Network styling - Default colors for neurons without specific styles
DEFAULT_NEURON_COLOR = 'lightblue'
DEFAULT_NEURON_EDGE_COLOR = 'navy'
DEFAULT_NEURON_EDGE_WIDTH = 1.5

# Connection line styling
CONNECTION_COLOR = 'gray'
CONNECTION_ALPHA = 0.4  # Transparency (0=invisible, 1=solid)
CONNECTION_LINEWIDTH = 1.5  # Line thickness

# Layer collapsing settings (for large layers like 300 neurons)
MAX_NEURONS_PER_LAYER = 8  # Collapse layers with more than this many neurons
COLLAPSE_NEURONS_START = 4  # Show this many neurons at the start
COLLAPSE_NEURONS_END = 4  # Show this many neurons at the end

# Color definitions for specific layers
COLOR_INPUT_NEURON = '#FFD700'  # Gold/yellow for input neurons
COLOR_INPUT_EDGE = '#B8860B'  # Dark goldenrod
COLOR_INPUT_BOX_FILL = '#FFFACD'  # Light yellow box
COLOR_INPUT_BOX_EDGE = '#B8860B'  # Dark goldenrod box edge

COLOR_OUTPUT1_NEURON = '#90EE90'  # Light green for output head 1
COLOR_OUTPUT1_EDGE = '#228B22'  # Forest green
COLOR_OUTPUT1_BOX_FILL = '#E6FFE6'  # Very light green box
COLOR_OUTPUT1_BOX_EDGE = '#228B22'  # Forest green box edge

COLOR_OUTPUT2_NEURON = '#FF6B6B'  # Light red/coral for output head 2
COLOR_OUTPUT2_EDGE = '#DC143C'  # Crimson
COLOR_OUTPUT2_BOX_FILL = '#FFE6E6'  # Very light red box
COLOR_OUTPUT2_BOX_EDGE = '#DC143C'  # Crimson box edge

# Box styling
BOX_EDGE_WIDTH = 2.5
BOX_PADDING = 0.7  # Space between neurons and box edge
BOX_CORNER_RADIUS = 0.4  # Roundness of box corners

# Output settings
OUTPUT_DPI_PDF = 600  # High resolution for publication PDFs
OUTPUT_DPI_PNG = 300  # Standard resolution for PNG presentations

# ==============================================================================
# END CONFIGURATION SECTION
# ==============================================================================

# Create output directory for figures
output_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("CEAS 2025 - Network Architecture Figures")
print("=" * 70)

# ==============================================================================
# Figure 1: Policy Network Architecture - Two-Head Output Network
# ==============================================================================
print("\n[Figure 1] Policy Network Architecture")
print("-" * 70)

# Create network with branching architecture (two output heads)
nn_main = NeuralNetwork("policy_network")

# Input layer: 6 neurons (yellow with box) with individual labels
input_layer = FullyConnectedLayer(
    num_neurons=6,
    name="Input",
    neuron_labels=[r"$\boldsymbol{q}_m$", r"$\dot{\boldsymbol{q}}_m$", r"$\tilde{\boldsymbol{r}}$", r"$\tilde{\boldsymbol{\theta}}$", r"$\tilde{\boldsymbol{v}}$", r"$\tilde{\boldsymbol{\omega}}$"],
    label_position="left"
)
nn_main.add_layer(input_layer)

# Hidden layer 1: 300 neurons
hidden1 = FullyConnectedLayer(
    num_neurons=300,
    activation="relu",
    name="Hidden_1"
)
nn_main.add_layer(hidden1, parent_ids=[input_layer.layer_id])

# Hidden layer 2: 300 neurons
hidden2 = FullyConnectedLayer(
    num_neurons=300,
    activation="relu",
    name="Hidden_2"
)
nn_main.add_layer(hidden2, parent_ids=[hidden1.layer_id])

# Hidden layer 3: 300 neurons
hidden3 = FullyConnectedLayer(
    num_neurons=300,
    activation="relu",
    name="Hidden_3"
)
nn_main.add_layer(hidden3, parent_ids=[hidden2.layer_id])

n = 7

# Output head 1 (top): 7 neurons - green with box with individual labels
output_head1 = FullyConnectedLayer(
    num_neurons=7,
    activation="softmax",
    name="Output_Head_1",
    neuron_labels=[fr"$\dot{{\phi}}_{i}$" for i in range(1, n+1)]
,
    label_position="right"
)
nn_main.add_layer(output_head1, parent_ids=[hidden3.layer_id])

# Output head 2 (bottom): 7 neurons - red with box with individual labels
output_head2 = FullyConnectedLayer(
    num_neurons=7,
    activation="softmax",
    name="Output_Head_2",
    neuron_labels=[fr"$\sigma_{{\dot{{\phi}}_{i}}}$" for i in range(1, n+1)]

,
    label_position="right"
)
nn_main.add_layer(output_head2, parent_ids=[hidden3.layer_id])

# Configure for publication quality using configuration variables
config_publication = PlotConfig(
    # Layout
    figsize=FIGSIZE,
    background_color=BACKGROUND_COLOR,
    
    # Font
    font_family='Times New Roman',
    
    # Display options
    show_title=SHOW_TITLE,
    show_layer_names=SHOW_LAYER_NAMES,
    show_neuron_labels=SHOW_NEURON_LABELS,
    neuron_numbering_reversed=NEURON_NUMBERING_REVERSED,
    
    # Neuron text labels
    show_neuron_text_labels=SHOW_NEURON_TEXT_LABELS,
    neuron_text_label_fontsize=NEURON_TEXT_LABEL_FONTSIZE,
    neuron_text_label_offset=NEURON_TEXT_LABEL_OFFSET,
    
    # Default neuron styling
    neuron_color=DEFAULT_NEURON_COLOR,
    neuron_edge_color=DEFAULT_NEURON_EDGE_COLOR,
    neuron_edge_width=DEFAULT_NEURON_EDGE_WIDTH,
    
    # Connection styling
    connection_color=CONNECTION_COLOR,
    connection_alpha=CONNECTION_ALPHA,
    connection_linewidth=CONNECTION_LINEWIDTH,
    
    # Layer collapsing
    max_neurons_per_layer=MAX_NEURONS_PER_LAYER,
    collapse_neurons_start=COLLAPSE_NEURONS_START,
    collapse_neurons_end=COLLAPSE_NEURONS_END,
    
    # Layer variable names (disabled - using individual neuron labels)
    show_layer_variable_names=False,
    
    # Layer-specific styles with boxes
    layer_styles={
        'Input': LayerStyle(
            neuron_fill_color=COLOR_INPUT_NEURON,
            neuron_edge_color=COLOR_INPUT_EDGE,
            neuron_edge_width=2.0,
            box_around_layer=True,
            box_fill_color=COLOR_INPUT_BOX_FILL,
            box_edge_color=COLOR_INPUT_BOX_EDGE,
            box_edge_width=BOX_EDGE_WIDTH,
            box_padding=BOX_PADDING,
            box_corner_radius=BOX_CORNER_RADIUS
        ),
        'Output_Head_1': LayerStyle(
            neuron_fill_color=COLOR_OUTPUT1_NEURON,
            neuron_edge_color=COLOR_OUTPUT1_EDGE,
            neuron_edge_width=2.0,
            box_around_layer=True,
            box_fill_color=COLOR_OUTPUT1_BOX_FILL,
            box_edge_color=COLOR_OUTPUT1_BOX_EDGE,
            box_edge_width=BOX_EDGE_WIDTH,
            box_padding=BOX_PADDING,
            box_corner_radius=BOX_CORNER_RADIUS
        ),
        'Output_Head_2': LayerStyle(
            neuron_fill_color=COLOR_OUTPUT2_NEURON,
            neuron_edge_color=COLOR_OUTPUT2_EDGE,
            neuron_edge_width=2.0,
            box_around_layer=True,
            box_fill_color=COLOR_OUTPUT2_BOX_FILL,
            box_edge_color=COLOR_OUTPUT2_BOX_EDGE,
            box_edge_width=BOX_EDGE_WIDTH,
            box_padding=BOX_PADDING,
            box_corner_radius=BOX_CORNER_RADIUS
        )
    }
)

# Generate PDF for publication (high DPI)
figure1_path = os.path.join(output_dir, "policy_net.pdf")
plot_network(
    nn_main,
    config=config_publication,
    save_path=figure1_path,
    show=False,
    dpi=OUTPUT_DPI_PDF,
    format="pdf"
)
print(f"✓ Generated: {figure1_path}")

# Also generate PNG for presentations/slides
figure1_png_path = os.path.join(output_dir, "policy_net.png")
plot_network(
    nn_main,
    config=config_publication,
    save_path=figure1_png_path,
    show=False,
    dpi=OUTPUT_DPI_PNG
)
print(f"✓ Generated: {figure1_png_path}")

# ==============================================================================
# Figure 2: Critic Network Architecture - Single Value Output
# ==============================================================================
print("\n[Figure 2] Critic Network Architecture")
print("-" * 70)

# Create critic network with same structure but single output
nn_critic = NeuralNetwork("critic_network")

# Input layer: 6 neurons (yellow with box) with individual labels
input_layer_critic = FullyConnectedLayer(
    num_neurons=6,
    name="Input",
    neuron_labels=[r"$\boldsymbol{q}_m$", r"$\dot{\boldsymbol{q}}_m$", r"$\tilde{\boldsymbol{r}}$", r"$\tilde{\boldsymbol{v}}$", r"$\tilde{\boldsymbol{\theta}}$", r"$\tilde{\boldsymbol{\omega}}$"],
    label_position="left"
)
nn_critic.add_layer(input_layer_critic)

# Hidden layer 1: 300 neurons
hidden1_critic = FullyConnectedLayer(
    num_neurons=300,
    activation="relu",
    name="Hidden_1"
)
nn_critic.add_layer(hidden1_critic, parent_ids=[input_layer_critic.layer_id])

# Hidden layer 2: 300 neurons
hidden2_critic = FullyConnectedLayer(
    num_neurons=300,
    activation="relu",
    name="Hidden_2"
)
nn_critic.add_layer(hidden2_critic, parent_ids=[hidden1_critic.layer_id])

# Hidden layer 3: 300 neurons
hidden3_critic = FullyConnectedLayer(
    num_neurons=300,
    activation="relu",
    name="Hidden_3"
)
nn_critic.add_layer(hidden3_critic, parent_ids=[hidden2_critic.layer_id])

# Output: 1 neuron - value function V(s)
output_critic = FullyConnectedLayer(
    num_neurons=1,
    activation="linear",
    name="Output",
    neuron_labels=[r"$V_\pi(s)$"],
    label_position="right"
)
nn_critic.add_layer(output_critic, parent_ids=[hidden3_critic.layer_id])

# Configure for publication quality using configuration variables
config_critic = PlotConfig(
    # Layout
    figsize=FIGSIZE,
    background_color=BACKGROUND_COLOR,
    
    # Font
    font_family='Times New Roman',
    
    # Display options
    show_title=SHOW_TITLE,
    show_layer_names=SHOW_LAYER_NAMES,
    show_neuron_labels=SHOW_NEURON_LABELS,
    neuron_numbering_reversed=NEURON_NUMBERING_REVERSED,
    
    # Neuron text labels
    show_neuron_text_labels=SHOW_NEURON_TEXT_LABELS,
    neuron_text_label_fontsize=NEURON_TEXT_LABEL_FONTSIZE,
    neuron_text_label_offset=NEURON_TEXT_LABEL_OFFSET,
    
    # Default neuron styling
    neuron_color=DEFAULT_NEURON_COLOR,
    neuron_edge_color=DEFAULT_NEURON_EDGE_COLOR,
    neuron_edge_width=DEFAULT_NEURON_EDGE_WIDTH,
    
    # Connection styling
    connection_color=CONNECTION_COLOR,
    connection_alpha=CONNECTION_ALPHA,
    connection_linewidth=CONNECTION_LINEWIDTH,
    
    # Layer collapsing
    max_neurons_per_layer=MAX_NEURONS_PER_LAYER,
    collapse_neurons_start=COLLAPSE_NEURONS_START,
    collapse_neurons_end=COLLAPSE_NEURONS_END,
    
    # Layer variable names (disabled - using individual neuron labels)
    show_layer_variable_names=False,
    
    # Layer-specific styles with boxes
    layer_styles={
        'Input': LayerStyle(
            neuron_fill_color=COLOR_INPUT_NEURON,
            neuron_edge_color=COLOR_INPUT_EDGE,
            neuron_edge_width=2.0,
            box_around_layer=True,
            box_fill_color=COLOR_INPUT_BOX_FILL,
            box_edge_color=COLOR_INPUT_BOX_EDGE,
            box_edge_width=BOX_EDGE_WIDTH,
            box_padding=BOX_PADDING,
            box_corner_radius=BOX_CORNER_RADIUS
        ),
        'Output': LayerStyle(
            neuron_fill_color=COLOR_OUTPUT1_NEURON,
            neuron_edge_color=COLOR_OUTPUT1_EDGE,
            neuron_edge_width=2.0,
            box_around_layer=True,
            box_fill_color=COLOR_OUTPUT1_BOX_FILL,
            box_edge_color=COLOR_OUTPUT1_BOX_EDGE,
            box_edge_width=BOX_EDGE_WIDTH,
            box_padding=BOX_PADDING,
            box_corner_radius=BOX_CORNER_RADIUS
        )
    }
)

# Generate PDF for publication (high DPI)
figure2_path = os.path.join(output_dir, "critic_net.pdf")
plot_network(
    nn_critic,
    config=config_critic,
    save_path=figure2_path,
    show=False,
    dpi=OUTPUT_DPI_PDF,
    format="pdf"
)
print(f"✓ Generated: {figure2_path}")

# Also generate PNG for presentations/slides
figure2_png_path = os.path.join(output_dir, "critic_net.png")
plot_network(
    nn_critic,
    config=config_critic,
    save_path=figure2_png_path,
    show=False,
    dpi=OUTPUT_DPI_PNG
)
print(f"✓ Generated: {figure2_png_path}")

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "=" * 70)
print("Figure Generation Complete!")
print("=" * 70)
print(f"\nAll figures saved to: {output_dir}/")
print("\nGenerated files:")
print("  - policy_net.pdf (publication)")
print("  - policy_net.png (presentation)")
print("  - critic_net.pdf (publication)")
print("  - critic_net.png (presentation)")
print("\nNetwork specifications:")
print("  [1] Policy Network:")
print("      - Input: 6 neurons (yellow with box)")
print("      - Hidden: 3 layers × 300 neurons (collapsed: 4 + dots + 4)")
print("      - Output Head 1: 7 neurons (green with box, top)")
print("      - Output Head 2: 7 neurons (red with box, bottom)")
print("  [2] Critic Network:")
print("      - Input: 6 neurons (yellow with box)")
print("      - Hidden: 3 layers × 300 neurons (collapsed: 4 + dots + 4)")
print("      - Output: 1 neuron V(s) (green with box)")
print("\nNext steps:")
print("  1. Review the generated figures")
print("  2. Adjust PlotConfig settings if needed")
print("  3. Add figure captions to your LaTeX document")
print("=" * 70)
