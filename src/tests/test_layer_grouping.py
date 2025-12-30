"""
Test script for layer grouping feature.

Tests:
1. Basic layer grouping with different bracket styles
2. Multiple groups in the same network
3. Customization of bracket and label properties
4. Encoder-Decoder architecture grouping
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer, VectorInput
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig, LayerGroup

# Create output directory at project root
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
output_dir = os.path.join(project_root, "test_outputs", "layer_grouping")
os.makedirs(output_dir, exist_ok=True)

def test_curly_brackets():
    """Test layer grouping with curly brackets."""
    print("\nTesting curly bracket style...")
    
    nn = NeuralNetwork("Curly Test")
    l1_id = nn.add_layer(VectorInput(num_features=5, name="L1"))
    l2_id = nn.add_layer(FullyConnectedLayer(5, activation="relu", name="L2"), parent_ids=[l1_id])
    nn.add_layer(FullyConnectedLayer(3, activation="softmax", name="L3"), parent_ids=[l2_id])
    
    config = PlotConfig(
        background_color='white',
        show_layer_names=True,
        layer_groups=[
            LayerGroup(
                layer_ids=["L1", "L2"],
                label="Group A",
                bracket_style='curly',
                bracket_color='blue',
                bracket_linewidth=2.0
            )
        ]
    )
    
    plot_network(
        nn,
        config=config,
        title="Curly Bracket Test",
        save_path=os.path.join(output_dir, "01_curly_bracket.png"),
        show=False
    )
    print("  ✓ Curly bracket grouping")


def test_square_brackets():
    """Test layer grouping with square brackets."""
    print("\nTesting square bracket style...")
    
    nn = NeuralNetwork("Square Test")
    l1_id = nn.add_layer(VectorInput(num_features=5, name="L1"))
    l2_id = nn.add_layer(FullyConnectedLayer(5, activation="relu", name="L2"), parent_ids=[l1_id])
    nn.add_layer(FullyConnectedLayer(3, activation="softmax", name="L3"), parent_ids=[l2_id])
    
    config = PlotConfig(
        background_color='white',
        show_layer_names=True,
        layer_groups=[
            LayerGroup(
                layer_ids=["L1", "L2"],
                label="Group B",
                bracket_style='square',
                bracket_color='green',
                bracket_linewidth=2.0
            )
        ]
    )
    
    plot_network(
        nn,
        config=config,
        title="Square Bracket Test",
        save_path=os.path.join(output_dir, "02_square_bracket.png"),
        show=False
    )
    print("  ✓ Square bracket grouping")


def test_straight_line():
    """Test layer grouping with straight line."""
    print("\nTesting straight line style...")
    
    nn = NeuralNetwork("Straight Test")
    l1_id = nn.add_layer(VectorInput(num_features=5, name="L1"))
    l2_id = nn.add_layer(FullyConnectedLayer(5, activation="relu", name="L2"), parent_ids=[l1_id])
    nn.add_layer(FullyConnectedLayer(3, activation="softmax", name="L3"), parent_ids=[l2_id])
    
    config = PlotConfig(
        background_color='white',
        show_layer_names=True,
        layer_groups=[
            LayerGroup(
                layer_ids=["L1", "L2"],
                label="Group D",
                bracket_style='straight',
                bracket_color='black',
                bracket_linewidth=2.5
            )
        ]
    )
    
    plot_network(
        nn,
        config=config,
        title="Straight Line Test",
        save_path=os.path.join(output_dir, "03_straight_line.png"),
        show=False
    )
    print("  ✓ Straight line grouping")


def test_multiple_groups():
    """Test multiple groups in the same network."""
    print("\nTesting multiple groups...")
    
    nn = NeuralNetwork("Multiple Groups")
    l1_id = nn.add_layer(VectorInput(num_features=5, name="Input"))
    l2_id = nn.add_layer(FullyConnectedLayer(8, activation="relu", name="Hidden1"), parent_ids=[l1_id])
    l3_id = nn.add_layer(FullyConnectedLayer(8, activation="relu", name="Hidden2"), parent_ids=[l2_id])
    l4_id = nn.add_layer(FullyConnectedLayer(5, activation="relu", name="Hidden3"), parent_ids=[l3_id])
    nn.add_layer(FullyConnectedLayer(3, activation="softmax", name="Output"), parent_ids=[l4_id])
    
    config = PlotConfig(
        background_color='white',
        show_layer_names=True,
        figsize=(14, 7),
        layer_groups=[
            LayerGroup(
                layer_ids=["Input", "Hidden1"],
                label="Stage 1",
                bracket_style='curly',
                bracket_color='blue',
                bracket_linewidth=2.0,
                label_color='blue'
            ),
            LayerGroup(
                layer_ids=["Hidden2", "Hidden3"],
                label="Stage 2",
                bracket_style='curly',
                bracket_color='green',
                bracket_linewidth=2.0,
                label_color='green'
            ),
            LayerGroup(
                layer_ids=["Output"],
                label="Stage 3",
                bracket_style='square',
                bracket_color='red',
                bracket_linewidth=2.0,
                label_color='red'
            )
        ]
    )
    
    plot_network(
        nn,
        config=config,
        title="Multiple Groups Test",
        save_path=os.path.join(output_dir, "04_multiple_groups.png"),
        show=False
    )
    print("  ✓ Multiple groups")


def test_encoder_decoder():
    """Test encoder-decoder architecture with grouping."""
    print("\nTesting encoder-decoder architecture...")
    
    nn = NeuralNetwork("Encoder-Decoder")
    input_id = nn.add_layer(VectorInput(num_features=10, name="Input"))
    enc1_id = nn.add_layer(FullyConnectedLayer(8, activation="relu", name="Enc1"), parent_ids=[input_id])
    enc2_id = nn.add_layer(FullyConnectedLayer(5, activation="relu", name="Enc2"), parent_ids=[enc1_id])
    latent_id = nn.add_layer(FullyConnectedLayer(3, activation="relu", name="Latent"), parent_ids=[enc2_id])
    dec1_id = nn.add_layer(FullyConnectedLayer(5, activation="relu", name="Dec1"), parent_ids=[latent_id])
    dec2_id = nn.add_layer(FullyConnectedLayer(8, activation="relu", name="Dec2"), parent_ids=[dec1_id])
    nn.add_layer(FullyConnectedLayer(10, activation="sigmoid", name="Output"), parent_ids=[dec2_id])
    
    config = PlotConfig(
        background_color='white',
        show_layer_names=True,
        figsize=(16, 7),
        layer_groups=[
            LayerGroup(
                layer_ids=["Input", "Enc1", "Enc2"],
                label="Encoder",
                bracket_style='curly',
                bracket_color='#1976D2',
                bracket_linewidth=2.5,
                label_fontsize=13,
                label_color='#1976D2'
            ),
            LayerGroup(
                layer_ids=["Latent"],
                label="Latent Space",
                bracket_style='square',
                bracket_color='#7B1FA2',
                bracket_linewidth=2.5,
                label_fontsize=13,
                label_color='#7B1FA2'
            ),
            LayerGroup(
                layer_ids=["Dec1", "Dec2", "Output"],
                label="Decoder",
                bracket_style='curly',
                bracket_color='#D32F2F',
                bracket_linewidth=2.5,
                label_fontsize=13,
                label_color='#D32F2F'
            )
        ]
    )
    
    plot_network(
        nn,
        config=config,
        title="Encoder-Decoder Architecture",
        save_path=os.path.join(output_dir, "05_encoder_decoder.png"),
        show=False,
        dpi=300
    )
    print("  ✓ Encoder-decoder architecture")


def test_customization():
    """Test customization options."""
    print("\nTesting customization options...")
    
    nn = NeuralNetwork("Customization Test")
    l1_id = nn.add_layer(VectorInput(num_features=6, name="Layer1"))
    l2_id = nn.add_layer(FullyConnectedLayer(6, activation="relu", name="Layer2"), parent_ids=[l1_id])
    nn.add_layer(FullyConnectedLayer(4, activation="softmax", name="Layer3"), parent_ids=[l2_id])
    
    config = PlotConfig(
        background_color='white',
        show_layer_names=True,
        layer_groups=[
            LayerGroup(
                layer_ids=["Layer1", "Layer2"],
                label="Custom Styled Group",
                bracket_style='curly',
                bracket_color='#FF6B35',
                bracket_linewidth=3.0,
                label_fontsize=15,
                label_color='#FF6B35',
                y_offset=-2.5,
                bracket_height=0.5
            )
        ]
    )
    
    plot_network(
        nn,
        config=config,
        title="Customization Options Test",
        save_path=os.path.join(output_dir, "06_customization.png"),
        show=False
    )
    print("  ✓ Customization options")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Layer Grouping Test Suite")
    print("=" * 60)
    
    test_curly_brackets()
    test_square_brackets()
    test_straight_line()
    test_multiple_groups()
    test_encoder_decoder()
    test_customization()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("Check 'test_outputs/layer_grouping' for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
