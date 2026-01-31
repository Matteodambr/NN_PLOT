"""
Test ImageInput layer functionality.
"""

import sys
import os

# Add the parent directory to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from NN_DEFINITION_UTILITIES import ImageInput, NeuralNetwork, FullyConnectedLayer
from NN_PLOTTING_UTILITIES import NetworkPlotter, PlotConfig
import matplotlib
matplotlib.rcParams['text.usetex'] = False  # Disable LaTeX rendering for tests


def test_image_input_text_mode():
    """Test ImageInput in text mode (no actual image)."""
    # Create network with ImageInput
    network = NeuralNetwork(name="Image CNN")
    
    # Add ImageInput layer with text mode
    img_input = ImageInput(
        height=224,
        width=224,
        channels=3,
        name="Input Image",
        display_mode="text",
        custom_text="224 x 224 x 3",  # Use plain text instead of LaTeX
        custom_text_size=14
    )
    network.add_layer(img_input, is_input=True)
    
    # Add a hidden layer
    hidden = FullyConnectedLayer(num_neurons=128, activation="ReLU", name="Hidden")
    network.add_layer(hidden, parent_ids=[img_input.layer_id])
    
    # Add output layer
    output = FullyConnectedLayer(num_neurons=10, activation="Softmax", name="Output")
    network.add_layer(output, parent_ids=[hidden.layer_id])
    
    # Create plotter and plot
    # Disable LaTeX after plotter creation to avoid LaTeX requirement
    import matplotlib as mpl
    config = PlotConfig(figsize=(10, 8))
    plotter = NetworkPlotter(config)
    mpl.rcParams['text.usetex'] = False  # Force disable after plotter sets it
    
    output_path = os.path.join(os.path.dirname(__file__), "../../PlottedNetworks/test_image_input_text.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plotter.plot_network(
        network,
        title="ImageInput - Text Mode",
        save_path=output_path,
        show=False,
        dpi=150
    )
    
    print(f"✓ Text mode test passed. Output saved to {output_path}")


def test_image_input_default_text():
    """Test ImageInput in text mode with default dimension text."""
    network = NeuralNetwork(name="Image CNN Default")
    
    # Add ImageInput layer without custom text (should show dimensions)
    img_input = ImageInput(
        height=128,
        width=128,
        channels=1,
        name="BW Input",
        display_mode="text"
    )
    network.add_layer(img_input, is_input=True)
    
    # Add output layer
    output = FullyConnectedLayer(num_neurons=10, activation="Softmax", name="Output")
    network.add_layer(output, parent_ids=[img_input.layer_id])
    
    # Create plotter and plot
    # Disable LaTeX after plotter creation
    import matplotlib as mpl
    config = PlotConfig(figsize=(8, 6))
    plotter = NetworkPlotter(config)
    mpl.rcParams['text.usetex'] = False  # Force disable after plotter sets it
    
    output_path = os.path.join(os.path.dirname(__file__), "../../PlottedNetworks/test_image_input_default.png")
    plotter.plot_network(
        network,
        title="ImageInput - Default Text",
        save_path=output_path,
        show=False,
        dpi=150
    )
    
    print(f"✓ Default text test passed. Output saved to {output_path}")


def test_image_input_validation():
    """Test ImageInput validation."""
    # Test invalid dimensions
    try:
        ImageInput(height=-1, width=224, channels=3)
        assert False, "Should have raised ValueError for negative height"
    except ValueError as e:
        print(f"✓ Validation test 1 passed: {e}")
    
    # Test invalid channels
    try:
        ImageInput(height=224, width=224, channels=2)
        assert False, "Should have raised ValueError for invalid channels"
    except ValueError as e:
        print(f"✓ Validation test 2 passed: {e}")
    
    # Test invalid display_mode
    try:
        ImageInput(height=224, width=224, channels=3, display_mode="invalid")
        assert False, "Should have raised ValueError for invalid display_mode"
    except ValueError as e:
        print(f"✓ Validation test 3 passed: {e}")
    
    # Test missing image_path for image mode
    try:
        ImageInput(height=224, width=224, channels=3, display_mode="single_image")
        assert False, "Should have raised ValueError for missing image_path"
    except ValueError as e:
        print(f"✓ Validation test 4 passed: {e}")
    
    # Test invalid magnification
    try:
        ImageInput(height=224, width=224, channels=3, magnification=-1)
        assert False, "Should have raised ValueError for negative magnification"
    except ValueError as e:
        print(f"✓ Validation test 5 passed: {e}")
    
    # Test invalid translation
    try:
        ImageInput(height=224, width=224, channels=3, translation_x=2.0)
        assert False, "Should have raised ValueError for out-of-range translation"
    except ValueError as e:
        print(f"✓ Validation test 6 passed: {e}")


def test_image_input_properties():
    """Test ImageInput properties and methods."""
    img_input = ImageInput(height=224, width=224, channels=3, name="Test")
    
    # Test get_output_size
    assert img_input.get_output_size() == 224 * 224 * 3, "Output size should be H×W×C"
    
    # Test num_neurons property
    assert img_input.num_neurons == 224 * 224 * 3, "num_neurons should match output size"
    
    # Test __str__
    str_repr = str(img_input)
    assert "224×224×3" in str_repr, "String representation should include dimensions"
    
    print("✓ Properties test passed")


if __name__ == "__main__":
    print("Running ImageInput tests...")
    print("-" * 50)
    
    # Run validation tests first
    test_image_input_validation()
    print()
    
    # Run property tests
    test_image_input_properties()
    print()
    
    # Run visualization tests
    test_image_input_text_mode()
    test_image_input_default_text()
    
    print("-" * 50)
    print("All tests passed! ✓")
