"""
Comprehensive test for ImageInput layer functionality.
Single test covering all display modes and features with actual images.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from src.NN_DEFINITION_UTILITIES import (
    NeuralNetwork,
    FullyConnectedLayer,
    ImageInput,
    NetworkType
)
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig


def get_catto_path():
    """Get the path to the repository's catto.jpg image."""
    # Path relative to this test file
    test_dir = os.path.dirname(os.path.abspath(__file__))
    catto_path = os.path.join(test_dir, '..', 'readme_image_static', 'catto.jpg')
    
    if not os.path.exists(catto_path):
        raise FileNotFoundError(f"catto.jpg not found at {catto_path}")
    
    return catto_path


def test_image_input_comprehensive():
    """Comprehensive test covering all ImageInput features."""
    print("\n" + "="*70)
    print("Comprehensive ImageInput Test - All Modes")
    print("="*70)
    
    # Get the cat image path
    cat_image_path = get_catto_path()
    print(f"Using cat image from: {cat_image_path}")
    
    # Get output directory (at project root, same as other tests)
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    output_dir = os.path.join(project_root, "test_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Test 1: Text mode with colored rectangle
    print("\n1. Testing text mode with colored rectangle...")
    nn1 = NeuralNetwork(name="Text Mode", network_type=NetworkType.FEEDFORWARD)
    img_input1 = ImageInput(
        height=224, width=224, channels=3,
        name="RGB Input",
        display_mode="text",
        custom_text="224×224×3\nRGB Input",
        color_mode="rgb",
        rounded_corners=True,
        custom_size=2.5
    )
    nn1.add_layer(img_input1, is_input=True)
    nn1.add_layer(FullyConnectedLayer(num_neurons=16, activation="relu", name="Conv1"))
    nn1.add_layer(FullyConnectedLayer(num_neurons=10, activation="softmax", name="Output"))
    
    plot_network(
        nn1,
        save_path=os.path.join(output_dir, "output_image_input_text.png"),
        show=False,
        config=PlotConfig(figsize=(8, 6))
    )
    print("   ✓ Saved: output_image_input_text.png")
    
    # Test 2: Black & white mode
    print("\n2. Testing BW image mode...")
    nn2 = NeuralNetwork(name="BW Mode", network_type=NetworkType.FEEDFORWARD)
    img_input2 = ImageInput(
        height=400, width=400, channels=1,
        name="BW Cat",
        display_mode="image",
        image_path=cat_image_path,
        color_mode="bw",
        rounded_corners=True,
        custom_size=2.5
    )
    nn2.add_layer(img_input2, is_input=True)
    nn2.add_layer(FullyConnectedLayer(num_neurons=16, activation="relu", name="Conv1"))
    nn2.add_layer(FullyConnectedLayer(num_neurons=10, activation="softmax", name="Output"))
    
    plot_network(
        nn2,
        save_path=os.path.join(output_dir, "output_image_input_bw.png"),
        show=False,
        config=PlotConfig(figsize=(8, 6))
    )
    print("   ✓ Saved: output_image_input_bw.png")
    
    # Test 3: RGB single image mode
    print("\n3. Testing RGB single image mode...")
    nn3 = NeuralNetwork(name="RGB Single", network_type=NetworkType.FEEDFORWARD)
    img_input3 = ImageInput(
        height=600, width=600, channels=3,
        name="RGB Cat",
        display_mode="image",
        image_path=cat_image_path,
        color_mode="rgb",
        separate_channels=False,
        rounded_corners=True,
        custom_size=2.5
    )
    nn3.add_layer(img_input3, is_input=True)
    nn3.add_layer(FullyConnectedLayer(num_neurons=16, activation="relu", name="Conv1"))
    nn3.add_layer(FullyConnectedLayer(num_neurons=10, activation="softmax", name="Output"))
    
    plot_network(
        nn3,
        save_path=os.path.join(output_dir, "output_image_input_rgb_single.png"),
        show=False,
        config=PlotConfig(figsize=(8, 6))
    )
    print("   ✓ Saved: output_image_input_rgb_single.png")
    
    # Test 4: RGB separated channels mode
    print("\n4. Testing RGB separated channels mode...")
    nn4 = NeuralNetwork(name="RGB Separated", network_type=NetworkType.FEEDFORWARD)
    img_input4 = ImageInput(
        height=300, width=300, channels=3,
        name="RGB Channels",
        display_mode="image",
        image_path=cat_image_path,
        color_mode="rgb",
        separate_channels=True,
        rounded_corners=True,
        custom_size=2.5
    )
    nn4.add_layer(img_input4, is_input=True)
    nn4.add_layer(FullyConnectedLayer(num_neurons=16, activation="relu", name="FC"))
    nn4.add_layer(FullyConnectedLayer(num_neurons=10, activation="softmax", name="Output"))
    
    plot_network(
        nn4,
        save_path=os.path.join(output_dir, "output_image_input_rgb_separated.png"),
        show=False,
        config=PlotConfig(figsize=(8, 6))
    )
    print("   ✓ Saved: output_image_input_rgb_separated.png")
    
    # Test 5: Multiple inputs with different sizes
    print("\n5. Testing multiple ImageInput layers with different sizes...")
    nn5 = NeuralNetwork(name="Multi-Input", network_type=NetworkType.FEEDFORWARD)
    
    img_small = ImageInput(
        height=64, width=64, channels=3,
        name="Small",
        display_mode="image",
        image_path=cat_image_path,
        color_mode="rgb",
        rounded_corners=True,
        custom_size=1.5
    )
    nn5.add_layer(img_small, is_input=True)
    
    img_medium = ImageInput(
        height=128, width=128, channels=3,
        name="Medium",
        display_mode="image",
        image_path=cat_image_path,
        color_mode="rgb",
        rounded_corners=True,
        custom_size=2.5
    )
    nn5.add_layer(img_medium, is_input=True)
    
    img_large = ImageInput(
        height=224, width=224, channels=3,
        name="Large",
        display_mode="image",
        image_path=cat_image_path,
        color_mode="rgb",
        rounded_corners=True,
        custom_size=3.5
    )
    nn5.add_layer(img_large, is_input=True)
    
    merge = FullyConnectedLayer(num_neurons=16, activation="relu", name="Merge")
    nn5.add_layer(merge, parent_ids=[img_small.layer_id, img_medium.layer_id, img_large.layer_id])
    
    output = FullyConnectedLayer(num_neurons=10, activation="softmax", name="Output")
    nn5.add_layer(output)  # Connects to merge automatically
    
    plot_network(
        nn5,
        save_path=os.path.join(output_dir, "output_image_input_multi.png"),
        show=False,
        config=PlotConfig(figsize=(10, 8))
    )
    print("   ✓ Saved: output_image_input_multi.png")
    
    print("\n" + "="*70)
    print("All ImageInput tests completed successfully!")
    print(f"Output location: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    try:
        test_image_input_comprehensive()
    except Exception as e:
        print(f"\nError running test: {e}")
        import traceback
        traceback.print_exc()
