"""
Test ImageInput layer with actual image display functionality.
"""

import sys
import os
import matplotlib
matplotlib.rcParams['text.usetex'] = False  # Disable LaTeX rendering for tests

# Add the parent directory to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from NN_DEFINITION_UTILITIES import ImageInput, NeuralNetwork, FullyConnectedLayer
from NN_PLOTTING_UTILITIES import NetworkPlotter, PlotConfig


def test_single_image_mode():
    """Test ImageInput with actual image display in image mode."""
    import matplotlib as mpl
    
    # Create network
    network = NeuralNetwork(name="CNN with Image")
    
    # Path to test image
    image_path = os.path.join(os.path.dirname(__file__), "test_images", "kitten_like.png")
    
    # Add ImageInput layer with image mode (RGB, single image)
    img_input = ImageInput(
        height=224,
        width=224,
        channels=3,
        name="Image Input",
        display_mode="image",
        image_path=image_path,
        color_mode="rgb",
        magnification=1.0,
        rounded_corners=True
    )
    network.add_layer(img_input, is_input=True)
    
    # Add Conv layer (represented as FC for now)
    conv = FullyConnectedLayer(num_neurons=64, activation="ReLU", name="Conv")
    network.add_layer(conv, parent_ids=[img_input.layer_id])
    
    # Add output layer
    output = FullyConnectedLayer(num_neurons=10, activation="Softmax", name="Output")
    network.add_layer(output, parent_ids=[conv.layer_id])
    
    # Create plotter and plot
    config = PlotConfig(figsize=(10, 8))
    plotter = NetworkPlotter(config)
    mpl.rcParams['text.usetex'] = False
    
    output_path = os.path.join(os.path.dirname(__file__), "../../PlottedNetworks/test_image_single.png")
    plotter.plot_network(
        network,
        title="ImageInput - Single Image Mode",
        save_path=output_path,
        show=False,
        dpi=150
    )
    
    print(f"✓ Single image mode test passed. Output saved to {output_path}")


def test_single_image_with_magnification():
    """Test ImageInput with magnification (zoom in)."""
    import matplotlib as mpl
    
    network = NeuralNetwork(name="CNN with Magnification")
    
    image_path = os.path.join(os.path.dirname(__file__), "test_images", "pattern.png")
    
    # Add ImageInput with magnification=1.5 (zoomed in)
    img_input = ImageInput(
        height=200,
        width=200,
        channels=3,
        name="Zoomed Input",
        display_mode="image",
        image_path=image_path,
        color_mode="rgb",
        magnification=1.5,
        translation_x=0.0,
        translation_y=0.0,
        rounded_corners=True
    )
    network.add_layer(img_input, is_input=True)
    
    output = FullyConnectedLayer(num_neurons=10, activation="Softmax", name="Output")
    network.add_layer(output, parent_ids=[img_input.layer_id])
    
    config = PlotConfig(figsize=(8, 6))
    plotter = NetworkPlotter(config)
    mpl.rcParams['text.usetex'] = False
    
    output_path = os.path.join(os.path.dirname(__file__), "../../PlottedNetworks/test_image_magnified.png")
    plotter.plot_network(
        network,
        title="ImageInput - Magnification 1.5x",
        save_path=output_path,
        show=False,
        dpi=150
    )
    
    print(f"✓ Magnification test passed. Output saved to {output_path}")


def test_single_image_with_translation():
    """Test ImageInput with translation offset."""
    import matplotlib as mpl
    
    network = NeuralNetwork(name="CNN with Translation")
    
    image_path = os.path.join(os.path.dirname(__file__), "test_images", "gradient.png")
    
    # Add ImageInput with translation (offset to upper-right)
    img_input = ImageInput(
        height=150,
        width=200,
        channels=3,
        name="Offset Input",
        display_mode="image",
        image_path=image_path,
        color_mode="rgb",
        magnification=1.2,
        translation_x=0.3,   # Move right
        translation_y=-0.3,  # Move up
        rounded_corners=True
    )
    network.add_layer(img_input, is_input=True)
    
    output = FullyConnectedLayer(num_neurons=10, activation="Softmax", name="Output")
    network.add_layer(output, parent_ids=[img_input.layer_id])
    
    config = PlotConfig(figsize=(8, 6))
    plotter = NetworkPlotter(config)
    mpl.rcParams['text.usetex'] = False
    
    output_path = os.path.join(os.path.dirname(__file__), "../../PlottedNetworks/test_image_translated.png")
    plotter.plot_network(
        network,
        title="ImageInput - Translation (0.3, -0.3)",
        save_path=output_path,
        show=False,
        dpi=150
    )
    
    print(f"✓ Translation test passed. Output saved to {output_path}")


def test_bw_conversion():
    """Test ImageInput with black and white conversion."""
    import matplotlib as mpl
    
    network = NeuralNetwork(name="CNN with BW")
    
    image_path = os.path.join(os.path.dirname(__file__), "test_images", "kitten_like.png")
    
    # Add ImageInput with BW conversion
    img_input = ImageInput(
        height=224,
        width=224,
        channels=1,
        name="BW Input",
        display_mode="image",
        image_path=image_path,
        color_mode="bw",
        rounded_corners=True
    )
    network.add_layer(img_input, is_input=True)
    
    output = FullyConnectedLayer(num_neurons=10, activation="Softmax", name="Output")
    network.add_layer(output, parent_ids=[img_input.layer_id])
    
    config = PlotConfig(figsize=(8, 6))
    plotter = NetworkPlotter(config)
    mpl.rcParams['text.usetex'] = False
    
    output_path = os.path.join(os.path.dirname(__file__), "../../PlottedNetworks/test_image_bw.png")
    plotter.plot_network(
        network,
        title="ImageInput - Black & White Mode",
        save_path=output_path,
        show=False,
        dpi=150
    )
    
    print(f"✓ BW conversion test passed. Output saved to {output_path}")


def test_rgb_channels_mode():
    """Test ImageInput with RGB channels separated into 3 overlapped rectangles."""
    import matplotlib as mpl
    
    network = NeuralNetwork(name="CNN with RGB Channels")
    
    image_path = os.path.join(os.path.dirname(__file__), "test_images", "gradient.png")
    
    # Add ImageInput with separate_channels=True
    img_input = ImageInput(
        height=150,
        width=200,
        channels=3,
        name="RGB Channels",
        display_mode="image",
        image_path=image_path,
        color_mode="rgb",
        separate_channels=True,
        rounded_corners=True
    )
    network.add_layer(img_input, is_input=True)
    
    output = FullyConnectedLayer(num_neurons=10, activation="Softmax", name="Output")
    network.add_layer(output, parent_ids=[img_input.layer_id])
    
    config = PlotConfig(figsize=(10, 6))
    plotter = NetworkPlotter(config)
    mpl.rcParams['text.usetex'] = False
    
    output_path = os.path.join(os.path.dirname(__file__), "../../PlottedNetworks/test_image_rgb_channels.png")
    plotter.plot_network(
        network,
        title="ImageInput - RGB Channels Separated",
        save_path=output_path,
        show=False,
        dpi=150
    )
    
    print(f"✓ RGB channels mode test passed. Output saved to {output_path}")


def test_sharp_corners():
    """Test ImageInput with sharp corners (no rounding)."""
    import matplotlib as mpl
    
    network = NeuralNetwork(name="CNN with Sharp Corners")
    
    image_path = os.path.join(os.path.dirname(__file__), "test_images", "pattern.png")
    
    # Add ImageInput with rounded_corners=False
    img_input = ImageInput(
        height=150,
        width=200,
        channels=3,
        name="Sharp Corners",
        display_mode="image",
        image_path=image_path,
        color_mode="rgb",
        rounded_corners=False
    )
    network.add_layer(img_input, is_input=True)
    
    output = FullyConnectedLayer(num_neurons=10, activation="Softmax", name="Output")
    network.add_layer(output, parent_ids=[img_input.layer_id])
    
    config = PlotConfig(figsize=(8, 6))
    plotter = NetworkPlotter(config)
    mpl.rcParams['text.usetex'] = False
    
    output_path = os.path.join(os.path.dirname(__file__), "../../PlottedNetworks/test_image_sharp_corners.png")
    plotter.plot_network(
        network,
        title="ImageInput - Sharp Corners",
        save_path=output_path,
        show=False,
        dpi=150
    )
    
    print(f"✓ Sharp corners test passed. Output saved to {output_path}")


if __name__ == "__main__":
    print("Running ImageInput image display tests...")
    print("-" * 50)
    
    # Check if test images exist
    test_images_dir = os.path.join(os.path.dirname(__file__), "test_images")
    if not os.path.exists(test_images_dir):
        print(f"Error: Test images directory not found at {test_images_dir}")
        print("Please run the test image creation script first.")
        sys.exit(1)
    
    # Run all tests
    test_single_image_mode()
    test_single_image_with_magnification()
    test_single_image_with_translation()
    test_bw_conversion()
    test_rgb_channels_mode()
    test_sharp_corners()
    
    print("-" * 50)
    print("All image display tests passed! ✓")
