"""
Comprehensive test of ImageInput layer with real cat images.
Tests all modes: text, BW, RGB single, and RGB separated channels.
Uses catto.jpg from the repository.
"""

import sys
import os
import matplotlib
matplotlib.rcParams['text.usetex'] = False

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from NN_DEFINITION_UTILITIES import ImageInput, NeuralNetwork, FullyConnectedLayer
from NN_PLOTTING_UTILITIES import NetworkPlotter, PlotConfig


def get_catto_path():
    """Get the path to catto.jpg from the repository."""
    # catto.jpg is in src/readme_image_static/
    catto_path = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "readme_image_static", 
        "catto.jpg"
    )
    catto_path = os.path.abspath(catto_path)
    
    if not os.path.exists(catto_path):
        raise FileNotFoundError(f"Could not find catto.jpg at {catto_path}")
    
    print(f"Using catto.jpg from: {catto_path}")
    return catto_path


def test_text_mode_with_color():
    """Test 1: Text mode with colored rounded rectangle."""
    import matplotlib as mpl
    
    network = NeuralNetwork(name="CNN Text Mode")
    
    # Text mode with custom styling
    img_input = ImageInput(
        height=224,
        width=224,
        channels=3,
        name="Input Image",
        display_mode="text",
        custom_text="224×224×3\nRGB Input",
        custom_text_size=12,
        rounded_corners=True
    )
    network.add_layer(img_input, is_input=True)
    
    # Add subsequent layers with fewer neurons
    conv1 = FullyConnectedLayer(num_neurons=16, activation="ReLU", name="Conv1")
    network.add_layer(conv1, parent_ids=[img_input.layer_id])
    
    pool1 = FullyConnectedLayer(num_neurons=8, activation="", name="Pool1")
    network.add_layer(pool1, parent_ids=[conv1.layer_id])
    
    output = FullyConnectedLayer(num_neurons=10, activation="Softmax", name="Output")
    network.add_layer(output, parent_ids=[pool1.layer_id])
    
    # Plot
    config = PlotConfig(figsize=(12, 8))
    plotter = NetworkPlotter(config)
    mpl.rcParams['text.usetex'] = False
    
    output_path = os.path.join(os.path.dirname(__file__), "../test_outputs/comprehensive_text_mode.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plotter.plot_network(
        network,
        title="ImageInput - Text Mode with Colored Rectangle",
        save_path=output_path,
        show=False,
        dpi=150
    )
    
    print(f"✓ Test 1: Text mode saved to {output_path}")


def test_bw_mode(cat_image_path):
    """Test 2: Black & white mode with real cat image."""
    import matplotlib as mpl
    
    network = NeuralNetwork(name="CNN BW Mode")
    
    # BW mode
    img_input = ImageInput(
        height=300,
        width=400,
        channels=1,
        name="BW Cat Input",
        display_mode="image",
        image_path=cat_image_path,
        color_mode="bw",
        rounded_corners=True
    )
    network.add_layer(img_input, is_input=True)
    
    # Add subsequent layers
    conv1 = FullyConnectedLayer(num_neurons=16, activation="ReLU", name="Conv1")
    network.add_layer(conv1, parent_ids=[img_input.layer_id])
    
    output = FullyConnectedLayer(num_neurons=10, activation="Softmax", name="Output")
    network.add_layer(output, parent_ids=[conv1.layer_id])
    
    # Plot
    config = PlotConfig(figsize=(12, 8))
    plotter = NetworkPlotter(config)
    mpl.rcParams['text.usetex'] = False
    
    output_path = os.path.join(os.path.dirname(__file__), "../test_outputs/comprehensive_bw_mode.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plotter.plot_network(
        network,
        title="ImageInput - Black & White Cat Image",
        save_path=output_path,
        show=False,
        dpi=150
    )
    
    print(f"✓ Test 2: BW mode saved to {output_path}")


def test_rgb_single_mode(cat_image_path):
    """Test 3: RGB mode with single image (full color cat)."""
    import matplotlib as mpl
    
    network = NeuralNetwork(name="CNN RGB Single")
    
    # RGB single mode
    img_input = ImageInput(
        height=300,
        width=400,
        channels=3,
        name="RGB Cat Input",
        display_mode="image",
        image_path=cat_image_path,
        color_mode="rgb",
        separate_channels=False,
        rounded_corners=True
    )
    network.add_layer(img_input, is_input=True)
    
    # Add subsequent layers
    conv1 = FullyConnectedLayer(num_neurons=16, activation="ReLU", name="Conv1")
    network.add_layer(conv1, parent_ids=[img_input.layer_id])
    
    output = FullyConnectedLayer(num_neurons=10, activation="Softmax", name="Output")
    network.add_layer(output, parent_ids=[conv1.layer_id])
    
    # Plot
    config = PlotConfig(figsize=(12, 8))
    plotter = NetworkPlotter(config)
    mpl.rcParams['text.usetex'] = False
    
    output_path = os.path.join(os.path.dirname(__file__), "../test_outputs/comprehensive_rgb_single.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plotter.plot_network(
        network,
        title="ImageInput - Full Color RGB Cat Image",
        save_path=output_path,
        show=False,
        dpi=150
    )
    
    print(f"✓ Test 3: RGB single mode saved to {output_path}")


def test_rgb_separated_mode(cat_image_path):
    """Test 4: RGB mode with separated channels (3 overlapped rectangles)."""
    import matplotlib as mpl
    
    network = NeuralNetwork(name="CNN RGB Separated")
    
    # RGB separated channels mode
    img_input = ImageInput(
        height=300,
        width=400,
        channels=3,
        name="RGB Channels",
        display_mode="image",
        image_path=cat_image_path,
        color_mode="rgb",
        separate_channels=True,
        rounded_corners=True
    )
    network.add_layer(img_input, is_input=True)
    
    # Add subsequent layers
    conv1 = FullyConnectedLayer(num_neurons=16, activation="ReLU", name="Conv1")
    network.add_layer(conv1, parent_ids=[img_input.layer_id])
    
    output = FullyConnectedLayer(num_neurons=10, activation="Softmax", name="Output")
    network.add_layer(output, parent_ids=[conv1.layer_id])
    
    # Plot
    config = PlotConfig(figsize=(12, 8))
    plotter = NetworkPlotter(config)
    mpl.rcParams['text.usetex'] = False
    
    output_path = os.path.join(os.path.dirname(__file__), "../test_outputs/comprehensive_rgb_separated.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plotter.plot_network(
        network,
        title="ImageInput - RGB Channels Separated (R, G, B)",
        save_path=output_path,
        show=False,
        dpi=150
    )
    
    print(f"✓ Test 4: RGB separated mode saved to {output_path}")


if __name__ == "__main__":
    print("Running Comprehensive ImageInput Tests with Real Cat Images")
    print("=" * 70)
    
    # Get catto.jpg path from repository
    cat_image_path = get_catto_path()
    print()
    
    # Run all tests
    test_text_mode_with_color()
    test_bw_mode(cat_image_path)
    test_rgb_single_mode(cat_image_path)
    test_rgb_separated_mode(cat_image_path)
    
    print("=" * 70)
    print("All comprehensive tests completed! ✓")
    print("\nGenerated visualizations:")
    print("  1. comprehensive_text_mode.png - Text mode with colored rectangle")
    print("  2. comprehensive_bw_mode.png - Black & white catto.jpg")
    print("  3. comprehensive_rgb_single.png - Full color RGB catto.jpg")
    print("  4. comprehensive_rgb_separated.png - RGB channels separated from catto.jpg")
