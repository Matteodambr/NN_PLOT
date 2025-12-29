"""
Demo script showing layer grouping with brackets and labels.

This script demonstrates:
1. Grouping layers with curly brackets
2. Grouping layers with square brackets
3. Grouping layers with round brackets
4. Grouping layers with straight lines
5. Multiple groups in the same network
6. Customizing bracket and label styles
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig, LayerGroup

# Create output directory
output_dir = "test_outputs"
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("Layer Grouping with Brackets Demo")
print("="*70)

# Example 1: Encoder-Decoder architecture with curly brackets
print("\n1. Creating Encoder-Decoder network with curly brackets...")
nn_enc_dec = NeuralNetwork("Encoder-Decoder")

input_layer = FullyConnectedLayer(10, name="Input")
input_id = nn_enc_dec.add_layer(input_layer)

enc1 = FullyConnectedLayer(8, activation="relu", name="Encoder_1")
enc1_id = nn_enc_dec.add_layer(enc1, parent_ids=[input_id])

enc2 = FullyConnectedLayer(5, activation="relu", name="Encoder_2")
enc2_id = nn_enc_dec.add_layer(enc2, parent_ids=[enc1_id])

latent = FullyConnectedLayer(3, activation="relu", name="Latent")
latent_id = nn_enc_dec.add_layer(latent, parent_ids=[enc2_id])

dec1 = FullyConnectedLayer(5, activation="relu", name="Decoder_1")
dec1_id = nn_enc_dec.add_layer(dec1, parent_ids=[latent_id])

dec2 = FullyConnectedLayer(8, activation="relu", name="Decoder_2")
dec2_id = nn_enc_dec.add_layer(dec2, parent_ids=[dec1_id])

output_layer = FullyConnectedLayer(10, activation="sigmoid", name="Output")
nn_enc_dec.add_layer(output_layer, parent_ids=[dec2_id])

config_enc_dec = PlotConfig(
    figsize=(14, 8),
    background_color='white',
    show_layer_names=True,
    layer_groups=[
        LayerGroup(
            layer_ids=["Input", "Encoder_1", "Encoder_2"],
            label="Encoder",
            bracket_style='curly',
            bracket_color='blue',
            bracket_linewidth=2.5,
            label_fontsize=14,
            label_color='blue',
            y_offset=-2.0,
            bracket_height=0.4,
            additional_spacing=1.0
        ),
        LayerGroup(
            layer_ids=["Latent"],
            label="Latent Space",
            bracket_style='curly',
            bracket_color='purple',
            bracket_linewidth=2.5,
            label_fontsize=14,
            label_color='purple',
            y_offset=-2.0,
            bracket_height=0.4,
            additional_spacing=1.0
        ),
        LayerGroup(
            layer_ids=["Decoder_1", "Decoder_2", "Output"],
            label="Decoder",
            bracket_style='curly',
            bracket_color='red',
            bracket_linewidth=2.5,
            label_fontsize=14,
            label_color='red',
            y_offset=-2.0,
            bracket_height=0.4,
            additional_spacing=1.0
        )
    ]
)

plot_network(
    nn_enc_dec,
    config=config_enc_dec,
    title="Autoencoder Architecture with Layer Grouping",
    save_path=os.path.join(output_dir, "demo_grouping_encoder_decoder.png"),
    show=False,
    dpi=300
)
print("✅ Created: test_outputs/demo_grouping_encoder_decoder.png")

# Example 2: Different bracket styles
print("\n2. Creating network with different bracket styles...")
nn_styles = NeuralNetwork("Bracket Styles")

l1_id = nn_styles.add_layer(FullyConnectedLayer(5, name="Layer_1"))
l2_id = nn_styles.add_layer(FullyConnectedLayer(5, activation="relu", name="Layer_2"), parent_ids=[l1_id])
l3_id = nn_styles.add_layer(FullyConnectedLayer(5, activation="relu", name="Layer_3"), parent_ids=[l2_id])
l4_id = nn_styles.add_layer(FullyConnectedLayer(5, activation="relu", name="Layer_4"), parent_ids=[l3_id])
nn_styles.add_layer(FullyConnectedLayer(3, activation="softmax", name="Layer_5"), parent_ids=[l4_id])

config_styles = PlotConfig(
    figsize=(16, 7),
    background_color='white',
    show_layer_names=True,
    layer_groups=[
        LayerGroup(
            layer_ids=["Layer_1", "Layer_2"],
            label="Curly Bracket",
            bracket_style='curly',
            bracket_color='darkgreen',
            bracket_linewidth=2.0,
            label_fontsize=11,
            label_color='darkgreen',
            y_offset=-1.8
        ),
        LayerGroup(
            layer_ids=["Layer_3"],
            label="Square Bracket",
            bracket_style='square',
            bracket_color='darkorange',
            bracket_linewidth=2.0,
            label_fontsize=11,
            label_color='darkorange',
            y_offset=-1.8
        ),
        LayerGroup(
            layer_ids=["Layer_4", "Layer_5"],
            label="Round Bracket",
            bracket_style='round',
            bracket_color='darkred',
            bracket_linewidth=2.0,
            label_fontsize=11,
            label_color='darkred',
            y_offset=-1.8
        )
    ]
)

plot_network(
    nn_styles,
    config=config_styles,
    title="Different Bracket Styles for Layer Grouping",
    save_path=os.path.join(output_dir, "demo_grouping_bracket_styles.png"),
    show=False,
    dpi=300
)
print("✅ Created: test_outputs/demo_grouping_bracket_styles.png")

# Example 3: CNN-style architecture with feature extraction and classification
print("\n3. Creating CNN-style architecture with grouped components...")
nn_cnn = NeuralNetwork("CNN Architecture")

conv1_id = nn_cnn.add_layer(FullyConnectedLayer(32, name="Conv1"))
conv2_id = nn_cnn.add_layer(FullyConnectedLayer(64, activation="relu", name="Conv2"), parent_ids=[conv1_id])
pool_id = nn_cnn.add_layer(FullyConnectedLayer(32, activation="relu", name="Pool"), parent_ids=[conv2_id])

fc1_id = nn_cnn.add_layer(FullyConnectedLayer(128, activation="relu", name="FC1"), parent_ids=[pool_id])
fc2_id = nn_cnn.add_layer(FullyConnectedLayer(64, activation="relu", name="FC2"), parent_ids=[fc1_id])
nn_cnn.add_layer(FullyConnectedLayer(10, activation="softmax", name="Output"), parent_ids=[fc2_id])

config_cnn = PlotConfig(
    figsize=(14, 7),
    background_color='white',
    show_layer_names=True,
    layer_groups=[
        LayerGroup(
            layer_ids=["Conv1", "Conv2", "Pool"],
            label="Feature Extraction",
            bracket_style='curly',
            bracket_color='#1976D2',
            bracket_linewidth=3.0,
            label_fontsize=13,
            label_color='#1976D2',
            y_offset=-2.2,
            bracket_height=0.5
        ),
        LayerGroup(
            layer_ids=["FC1", "FC2", "Output"],
            label="Classification",
            bracket_style='curly',
            bracket_color='#D32F2F',
            bracket_linewidth=3.0,
            label_fontsize=13,
            label_color='#D32F2F',
            y_offset=-2.2,
            bracket_height=0.5
        )
    ]
)

plot_network(
    nn_cnn,
    config=config_cnn,
    title="CNN-Style Architecture: Feature Extraction + Classification",
    save_path=os.path.join(output_dir, "demo_grouping_cnn_style.png"),
    show=False,
    dpi=300
)
print("✅ Created: test_outputs/demo_grouping_cnn_style.png")

# Example 4: Simple example with straight line style
print("\n4. Creating simple network with straight line grouping...")
nn_simple = NeuralNetwork("Simple Grouping")

i_id = nn_simple.add_layer(FullyConnectedLayer(4, name="Input"))
h1_id = nn_simple.add_layer(FullyConnectedLayer(6, activation="relu", name="Hidden1"), parent_ids=[i_id])
h2_id = nn_simple.add_layer(FullyConnectedLayer(6, activation="relu", name="Hidden2"), parent_ids=[h1_id])
nn_simple.add_layer(FullyConnectedLayer(3, activation="softmax", name="Output"), parent_ids=[h2_id])

config_simple = PlotConfig(
    figsize=(12, 6),
    background_color='white',
    show_layer_names=True,
    layer_groups=[
        LayerGroup(
            layer_ids=["Hidden1", "Hidden2"],
            label="Processing Layers",
            bracket_style='straight',
            bracket_color='black',
            bracket_linewidth=2.5,
            label_fontsize=12,
            label_color='black',
            y_offset=-1.5
        )
    ]
)

plot_network(
    nn_simple,
    config=config_simple,
    title="Simple Network with Straight Line Grouping",
    save_path=os.path.join(output_dir, "demo_grouping_simple.png"),
    show=False,
    dpi=300
)
print("✅ Created: test_outputs/demo_grouping_simple.png")

print("\n" + "="*70)
print("Demo complete!")
print("="*70)
print("\nLayer Grouping Features:")
print("  • Group multiple layers with brackets and labels")
print("  • Bracket styles: 'curly', 'square', 'round', 'straight'")
print("  • Full customization of bracket color and line width")
print("  • Full customization of label color and font size")
print("  • Adjustable positioning (y_offset, bracket_height)")
print("  • additional_spacing: controls gap from layer labels")
print("\nExample Usage:")
print("  config = PlotConfig(")
print("      layer_groups=[")
print("          LayerGroup(")
print("              layer_ids=['Layer1', 'Layer2', 'Layer3'],")
print("              label='Encoder',")
print("              bracket_style='curly',")
print("              bracket_color='blue',")
print("              bracket_linewidth=2.5,")
print("              label_fontsize=14,")
print("              label_color='blue',")
print("              additional_spacing=1.0  # Gap from layer labels")
print("          )")
print("      ]")
print("  )")
print("="*70)
