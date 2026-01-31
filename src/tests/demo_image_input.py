"""
Comprehensive demo of ImageInput layer functionality.

This demo showcases all the features of the ImageInput layer:
1. Text mode with custom text
2. Single image mode with actual images
3. Magnification and translation
4. Black & white conversion
5. RGB channel separation
6. Rounded vs sharp corners
"""

import sys
import os
import matplotlib
matplotlib.rcParams['text.usetex'] = False

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from NN_DEFINITION_UTILITIES import ImageInput, NeuralNetwork, FullyConnectedLayer
from NN_PLOTTING_UTILITIES import NetworkPlotter, PlotConfig


def demo_all_modes():
    """Demonstrate all ImageInput display modes in a single network."""
    import matplotlib as mpl
    
    print("Creating comprehensive CNN architecture demo...")
    
    network = NeuralNetwork(name="CNN Architecture Demo")
    
    # Path to test image
    image_path = os.path.join(os.path.dirname(__file__), "test_images", "kitten_like.png")
    
    # Example 1: Text mode (default)
    img_text = ImageInput(
        height=224,
        width=224,
        channels=3,
        name="Text Mode",
        display_mode="text",
        custom_text="224 x 224 x 3"
    )
    network.add_layer(img_text, is_input=True)
    
    # Add first conv layer
    conv1 = FullyConnectedLayer(num_neurons=64, activation="ReLU", name="Conv1")
    network.add_layer(conv1, parent_ids=[img_text.layer_id])
    
    # Add pooling (represented as FC)
    pool1 = FullyConnectedLayer(num_neurons=32, activation="", name="Pool1")
    network.add_layer(pool1, parent_ids=[conv1.layer_id])
    
    # Add second conv layer
    conv2 = FullyConnectedLayer(num_neurons=128, activation="ReLU", name="Conv2")
    network.add_layer(conv2, parent_ids=[pool1.layer_id])
    
    # Add final layers
    flatten = FullyConnectedLayer(num_neurons=256, activation="", name="Flatten")
    network.add_layer(flatten, parent_ids=[conv2.layer_id])
    
    fc = FullyConnectedLayer(num_neurons=128, activation="ReLU", name="FC")
    network.add_layer(fc, parent_ids=[flatten.layer_id])
    
    output = FullyConnectedLayer(num_neurons=10, activation="Softmax", name="Output")
    network.add_layer(output, parent_ids=[fc.layer_id])
    
    # Plot
    config = PlotConfig(figsize=(12, 10))
    plotter = NetworkPlotter(config)
    mpl.rcParams['text.usetex'] = False
    
    output_path = os.path.join(os.path.dirname(__file__), "../../PlottedNetworks/demo_cnn_architecture.png")
    plotter.plot_network(
        network,
        title="CNN Architecture with ImageInput",
        save_path=output_path,
        show=False,
        dpi=150
    )
    
    print(f"✓ CNN demo saved to {output_path}")


def demo_comparison():
    """Create a comparison showing text vs image vs RGB channels modes."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    print("Creating side-by-side comparison...")
    
    image_path = os.path.join(os.path.dirname(__file__), "test_images", "gradient.png")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    mpl.rcParams['text.usetex'] = False
    
    # 1. Text mode
    network1 = NeuralNetwork(name="Text Mode")
    img1 = ImageInput(
        height=150, width=200, channels=3,
        name="Text",
        display_mode="text",
        custom_text="150 x 200 x 3"
    )
    network1.add_layer(img1, is_input=True)
    out1 = FullyConnectedLayer(num_neurons=10, activation="Softmax", name="Output")
    network1.add_layer(out1, parent_ids=[img1.layer_id])
    
    config1 = PlotConfig(figsize=(6, 6))
    plotter1 = NetworkPlotter(config1)
    plotter1.plot_network(network1, title="Text Mode", ax=axes[0], show=False)
    
    # 2. Single image mode
    network2 = NeuralNetwork(name="Single Image")
    img2 = ImageInput(
        height=150, width=200, channels=3,
        name="Single Image",
        display_mode="single_image",
        image_path=image_path
    )
    network2.add_layer(img2, is_input=True)
    out2 = FullyConnectedLayer(num_neurons=10, activation="Softmax", name="Output")
    network2.add_layer(out2, parent_ids=[img2.layer_id])
    
    config2 = PlotConfig(figsize=(6, 6))
    plotter2 = NetworkPlotter(config2)
    plotter2.plot_network(network2, title="Single Image Mode", ax=axes[1], show=False)
    
    # 3. RGB channels mode
    network3 = NeuralNetwork(name="RGB Channels")
    img3 = ImageInput(
        height=150, width=200, channels=3,
        name="RGB Channels",
        display_mode="rgb_channels",
        image_path=image_path
    )
    network3.add_layer(img3, is_input=True)
    out3 = FullyConnectedLayer(num_neurons=10, activation="Softmax", name="Output")
    network3.add_layer(out3, parent_ids=[img3.layer_id])
    
    config3 = PlotConfig(figsize=(6, 6))
    plotter3 = NetworkPlotter(config3)
    plotter3.plot_network(network3, title="RGB Channels Mode", ax=axes[2], show=False)
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "../../PlottedNetworks/demo_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison demo saved to {output_path}")


def demo_image_transforms():
    """Demonstrate magnification and translation features."""
    import matplotlib as mpl
    
    print("Creating image transformation demo...")
    
    image_path = os.path.join(os.path.dirname(__file__), "test_images", "pattern.png")
    
    network = NeuralNetwork(name="Image Transforms")
    
    # Add ImageInput with magnification and translation
    img_input = ImageInput(
        height=150,
        width=200,
        channels=3,
        name="Transformed Image",
        display_mode="single_image",
        image_path=image_path,
        magnification=2.0,      # 2x zoom
        translation_x=0.25,     # Shift right
        translation_y=-0.25,    # Shift up
        rounded_corners=True
    )
    network.add_layer(img_input, is_input=True)
    
    # Add processing layers
    conv = FullyConnectedLayer(num_neurons=32, activation="ReLU", name="Process")
    network.add_layer(conv, parent_ids=[img_input.layer_id])
    
    output = FullyConnectedLayer(num_neurons=10, activation="Softmax", name="Output")
    network.add_layer(output, parent_ids=[conv.layer_id])
    
    # Plot
    config = PlotConfig(figsize=(10, 8))
    plotter = NetworkPlotter(config)
    mpl.rcParams['text.usetex'] = False
    
    output_path = os.path.join(os.path.dirname(__file__), "../../PlottedNetworks/demo_transforms.png")
    plotter.plot_network(
        network,
        title="ImageInput with Magnification (2x) and Translation",
        save_path=output_path,
        show=False,
        dpi=150
    )
    
    print(f"✓ Transform demo saved to {output_path}")


def print_usage_examples():
    """Print code examples for using ImageInput."""
    print("\n" + "="*70)
    print("ImageInput Usage Examples")
    print("="*70)
    
    print("""
1. Basic Text Mode (no image file needed):
   
   img_input = ImageInput(
       height=224, width=224, channels=3,
       name="Input Image",
       display_mode="text",
       custom_text="224 x 224 x 3"
   )

2. Display Actual Image:
   
   img_input = ImageInput(
       height=224, width=224, channels=3,
       name="Photo Input",
       display_mode="single_image",
       image_path="path/to/image.jpg",  # or URL
       rounded_corners=True
   )

3. Magnification (Zoom In):
   
   img_input = ImageInput(
       height=224, width=224, channels=3,
       display_mode="single_image",
       image_path="image.jpg",
       magnification=1.5,  # 1.5x zoom
       translation_x=0.2,   # optional: shift right
       translation_y=-0.1   # optional: shift up
   )

4. Black & White Conversion:
   
   img_input = ImageInput(
       height=224, width=224, channels=1,
       display_mode="single_image",
       image_path="color_image.jpg",
       color_mode="bw"  # converts to grayscale
   )

5. RGB Channel Separation (3 overlapped rectangles):
   
   img_input = ImageInput(
       height=224, width=224, channels=3,
       display_mode="rgb_channels",
       image_path="image.jpg"
   )

6. Sharp Corners (no rounding):
   
   img_input = ImageInput(
       height=224, width=224, channels=3,
       display_mode="single_image",
       image_path="image.jpg",
       rounded_corners=False
   )
""")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("Running ImageInput comprehensive demos...")
    print("-" * 70)
    
    # Print usage examples
    print_usage_examples()
    
    # Check if test images exist
    test_images_dir = os.path.join(os.path.dirname(__file__), "test_images")
    if not os.path.exists(test_images_dir):
        print(f"Error: Test images directory not found at {test_images_dir}")
        print("Please run the test image creation script first.")
        sys.exit(1)
    
    # Run demos
    demo_all_modes()
    demo_image_transforms()
    demo_comparison()
    
    print("-" * 70)
    print("All demos completed successfully! ✓")
    print("\nGenerated images:")
    print("  - PlottedNetworks/demo_cnn_architecture.png")
    print("  - PlottedNetworks/demo_transforms.png")
    print("  - PlottedNetworks/demo_comparison.png")
