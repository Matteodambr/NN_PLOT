"""
Test script demonstrating custom neuron text labels with LaTeX support.

This script showcases:
1. Plain text labels for neurons
2. LaTeX math labels for neurons  
3. Mixed text and LaTeX labels
4. Left/right positioning of labels
5. Show/hide control for labels
6. Integration with existing features (numbering, collapsing)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer, VectorInput
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig

# Create output directory at project root
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
output_dir = os.path.join(project_root, "test_outputs", "neuron_labels")
os.makedirs(output_dir, exist_ok=True)

def test_basic_text_labels():
    """Test basic plain text labels on neurons."""
    print("Testing basic plain text labels...")
    
    nn = NeuralNetwork("Basic Labels Network")
    
    # Input layer with text labels on the left
    nn.add_layer(VectorInput(
        num_features=3,
        name="Input",
        neuron_labels=["Age", "Income", "Credit Score"],
        label_position="left"
    ))
    
    nn.add_layer(FullyConnectedLayer(
        num_neurons=4,
        activation="relu",
        name="Hidden"
    ))
    
    # Output layer with labels on the right
    nn.add_layer(FullyConnectedLayer(
        num_neurons=2,
        activation="softmax",
        name="Output",
        neuron_labels=["Approved", "Denied"],
        label_position="right"
    ))
    
    config = PlotConfig(show_neuron_text_labels=True)
    plot_network(
        nn,
        config=config,
        title="Plain Text Labels",
        save_path=os.path.join(output_dir, "01_basic_text.png"),
        show=False
    )
    print("  ✓ Basic text labels")


def test_latex_math_labels():
    """Test LaTeX mathematical notation labels."""
    print("\nTesting LaTeX math labels...")
    
    nn = NeuralNetwork("LaTeX Math Network")
    
    # Input layer with LaTeX math
    nn.add_layer(VectorInput(
        num_features=3,
        name="Input",
        neuron_labels=[r"$x_1$", r"$x_2$", r"$x_3$"],
        label_position="left"
    ))
    
    nn.add_layer(FullyConnectedLayer(
        num_neurons=5,
        activation="tanh",
        name="Hidden",
        neuron_labels=[r"$h_1$", r"$h_2$", r"$h_3$", r"$h_4$", r"$h_5$"],
        label_position="left"
    ))
    
    # Output with LaTeX
    nn.add_layer(FullyConnectedLayer(
        num_neurons=2,
        activation="sigmoid",
        name="Output",
        neuron_labels=[r"$\hat{y}_0$", r"$\hat{y}_1$"],
        label_position="right"
    ))
    
    config = PlotConfig(
        show_neuron_text_labels=True,
        neuron_text_label_fontsize=12
    )
    plot_network(
        nn,
        config=config,
        title="LaTeX Mathematical Notation",
        save_path=os.path.join(output_dir, "02_latex_math.png"),
        show=False
    )
    print("  ✓ LaTeX math labels")


def test_complex_latex():
    """Test complex LaTeX expressions."""
    print("\nTesting complex LaTeX expressions...")
    
    nn = NeuralNetwork("Complex LaTeX")
    
    # Input with complex LaTeX
    nn.add_layer(VectorInput(
        num_features=4,
        name="Input",
        neuron_labels=[
            r"$\alpha$",
            r"$\beta^2$",
            r"$\frac{a}{b}$",
            r"$\sum_{i=1}^n x_i$"
        ],
        label_position="left"
    ))
    
    nn.add_layer(FullyConnectedLayer(
        num_neurons=3,
        activation="relu",
        name="Hidden"
    ))
    
    # Output with Greek letters and symbols
    nn.add_layer(FullyConnectedLayer(
        num_neurons=3,
        activation="softmax",
        name="Output",
        neuron_labels=[r"$\theta$", r"$\phi$", r"$\psi$"],
        label_position="right"
    ))
    
    config = PlotConfig(
        show_neuron_text_labels=True,
        neuron_text_label_fontsize=11
    )
    plot_network(
        nn,
        config=config,
        title="Complex LaTeX Expressions",
        save_path=os.path.join(output_dir, "03_complex_latex.png"),
        show=False
    )
    print("  ✓ Complex LaTeX expressions")


def test_left_and_right_positioning():
    """Test labels on left vs right sides."""
    print("\nTesting left/right positioning...")
    
    # Network with labels on left
    nn_left = NeuralNetwork("Left Labels")
    nn_left.add_layer(VectorInput(
        num_features=4,
        name="Input",
        neuron_labels=[r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$"],
        label_position="left"
    ))
    nn_left.add_layer(FullyConnectedLayer(3, activation="relu", name="Hidden"))
    nn_left.add_layer(FullyConnectedLayer(2, activation="softmax", name="Output"))
    
    config = PlotConfig(show_neuron_text_labels=True)
    plot_network(
        nn_left,
        config=config,
        title="Labels on Left Side",
        save_path=os.path.join(output_dir, "04_labels_left.png"),
        show=False
    )
    print("  ✓ Labels on left")
    
    # Network with labels on right
    nn_right = NeuralNetwork("Right Labels")
    nn_right.add_layer(VectorInput(num_features=4, name="Input"))
    nn_right.add_layer(FullyConnectedLayer(3, activation="relu", name="Hidden"))
    nn_right.add_layer(FullyConnectedLayer(
        num_neurons=2,
        activation="softmax",
        name="Output",
        neuron_labels=[r"$\hat{y}_1$", r"$\hat{y}_2$"],
        label_position="right"
    ))
    
    plot_network(
        nn_right,
        config=config,
        title="Labels on Right Side",
        save_path=os.path.join(output_dir, "05_labels_right.png"),
        show=False
    )
    print("  ✓ Labels on right")


def test_mixed_labels():
    """Test network with both labeled and unlabeled layers."""
    print("\nTesting mixed labeled/unlabeled layers...")
    
    nn = NeuralNetwork("Mixed Labels")
    
    # Input with labels
    nn.add_layer(VectorInput(
        num_features=3,
        name="Input",
        neuron_labels=["Feature A", "Feature B", "Feature C"],
        label_position="left"
    ))
    
    # Hidden without labels
    nn.add_layer(FullyConnectedLayer(
        num_neurons=5,
        activation="relu",
        name="Hidden 1"
    ))
    
    # Another hidden without labels
    nn.add_layer(FullyConnectedLayer(
        num_neurons=4,
        activation="relu",
        name="Hidden 2"
    ))
    
    # Output with labels
    nn.add_layer(FullyConnectedLayer(
        num_neurons=3,
        activation="softmax",
        name="Output",
        neuron_labels=["Class A", "Class B", "Class C"],
        label_position="right"
    ))
    
    config = PlotConfig(show_neuron_text_labels=True)
    plot_network(
        nn,
        config=config,
        title="Mixed Labeled and Unlabeled Layers",
        save_path=os.path.join(output_dir, "06_mixed_labels.png"),
        show=False
    )
    print("  ✓ Mixed labeled/unlabeled layers")


def test_show_hide_control():
    """Test showing and hiding custom labels."""
    print("\nTesting show/hide control...")
    
    nn = NeuralNetwork("Show/Hide Test")
    nn.add_layer(VectorInput(
        num_features=3,
        name="Input",
        neuron_labels=[r"$x_1$", r"$x_2$", r"$x_3$"],
        label_position="left"
    ))
    nn.add_layer(FullyConnectedLayer(4, activation="relu", name="Hidden"))
    nn.add_layer(FullyConnectedLayer(
        num_neurons=2,
        activation="softmax",
        name="Output",
        neuron_labels=[r"$y_1$", r"$y_2$"],
        label_position="right"
    ))
    
    # With labels shown
    config_show = PlotConfig(show_neuron_text_labels=True)
    plot_network(
        nn,
        config=config_show,
        title="Custom Labels Shown",
        save_path=os.path.join(output_dir, "07_labels_shown.png"),
        show=False
    )
    print("  ✓ Labels shown")
    
    # With labels hidden
    config_hide = PlotConfig(show_neuron_text_labels=False)
    plot_network(
        nn,
        config=config_hide,
        title="Custom Labels Hidden",
        save_path=os.path.join(output_dir, "08_labels_hidden.png"),
        show=False
    )
    print("  ✓ Labels hidden")


def test_with_neuron_indices():
    """Test custom labels combined with neuron index numbers."""
    print("\nTesting custom labels with neuron indices...")
    
    nn = NeuralNetwork("Labels + Indices")
    nn.add_layer(VectorInput(
        num_features=4,
        name="Input",
        neuron_labels=[r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$"],
        label_position="left"
    ))
    nn.add_layer(FullyConnectedLayer(3, activation="relu", name="Hidden"))
    nn.add_layer(FullyConnectedLayer(
        num_neurons=2,
        activation="softmax",
        name="Output",
        neuron_labels=["Positive", "Negative"],
        label_position="right"
    ))
    
    config = PlotConfig(
        show_neuron_labels=True,  # Show indices
        show_neuron_text_labels=True  # Show custom text
    )
    plot_network(
        nn,
        config=config,
        title="Custom Labels + Neuron Indices",
        save_path=os.path.join(output_dir, "09_labels_and_indices.png"),
        show=False
    )
    print("  ✓ Custom labels with indices")


def test_with_collapsed_layers():
    """Test custom labels on collapsed layers."""
    print("\nTesting labels with collapsed layers...")
    
    nn = NeuralNetwork("Collapsed + Labels")
    
    # Large input layer with labels
    input_labels = [f"Feature {i+1}" for i in range(20)]
    nn.add_layer(VectorInput(
        num_features=20,
        name="Input",
        neuron_labels=input_labels,
        label_position="left"
    ))
    
    nn.add_layer(FullyConnectedLayer(
        num_neurons=15,
        activation="relu",
        name="Hidden"
    ))
    
    # Output with labels
    nn.add_layer(FullyConnectedLayer(
        num_neurons=5,
        activation="softmax",
        name="Output",
        neuron_labels=[f"Class {i}" for i in range(5)],
        label_position="right"
    ))
    
    config = PlotConfig(
        show_neuron_text_labels=True,
        max_neurons_per_layer=12,
        collapse_neurons_start=5,
        collapse_neurons_end=5
    )
    plot_network(
        nn,
        config=config,
        title="Custom Labels on Collapsed Layers",
        save_path=os.path.join(output_dir, "10_collapsed_with_labels.png"),
        show=False
    )
    print("  ✓ Labels with collapsed layers")


def test_font_sizes():
    """Test different font sizes for labels."""
    print("\nTesting different font sizes...")
    
    nn = NeuralNetwork("Font Size Test")
    nn.add_layer(VectorInput(
        num_features=3,
        name="Input",
        neuron_labels=[r"$\alpha$", r"$\beta$", r"$\gamma$"],
        label_position="left"
    ))
    nn.add_layer(FullyConnectedLayer(3, activation="relu", name="Hidden"))
    nn.add_layer(FullyConnectedLayer(
        num_neurons=2,
        activation="softmax",
        name="Output",
        neuron_labels=["Yes", "No"],
        label_position="right"
    ))
    
    # Small font
    config_small = PlotConfig(
        show_neuron_text_labels=True,
        neuron_text_label_fontsize=8
    )
    plot_network(
        nn,
        config=config_small,
        title="Small Font (8pt)",
        save_path=os.path.join(output_dir, "11_font_small.png"),
        show=False
    )
    print("  ✓ Small font (8pt)")
    
    # Large font
    config_large = PlotConfig(
        show_neuron_text_labels=True,
        neuron_text_label_fontsize=14
    )
    plot_network(
        nn,
        config=config_large,
        title="Large Font (14pt)",
        save_path=os.path.join(output_dir, "12_font_large.png"),
        show=False
    )
    print("  ✓ Large font (14pt)")


def test_realistic_example():
    """Test a realistic machine learning example."""
    print("\nTesting realistic ML example...")
    
    nn = NeuralNetwork("Credit Risk Predictor")
    
    # Input features
    nn.add_layer(VectorInput(
        num_features=6,
        name="Input Features",
        neuron_labels=[
            "Age",
            "Income",
            "Credit Score",
            "Debt Ratio",
            "Employment Years",
            "Loan Amount"
        ],
        label_position="left"
    ))
    
    nn.add_layer(FullyConnectedLayer(
        num_neurons=8,
        activation="relu",
        name="Hidden Layer 1"
    ))
    
    nn.add_layer(FullyConnectedLayer(
        num_neurons=4,
        activation="relu",
        name="Hidden Layer 2"
    ))
    
    # Output predictions
    nn.add_layer(FullyConnectedLayer(
        num_neurons=3,
        activation="softmax",
        name="Risk Category",
        neuron_labels=[
            "Low Risk",
            "Medium Risk",
            "High Risk"
        ],
        label_position="right"
    ))
    
    config = PlotConfig(
        show_neuron_text_labels=True,
        neuron_text_label_fontsize=10,
        figsize=(14, 8)
    )
    plot_network(
        nn,
        config=config,
        title="Credit Risk Prediction Neural Network",
        save_path=os.path.join(output_dir, "13_realistic_example.png"),
        show=False,
        dpi=300
    )
    
    # Also save as SVG for presentations
    plot_network(
        nn,
        config=config,
        title="Credit Risk Prediction Neural Network",
        save_path=os.path.join(output_dir, "14_realistic_example.svg"),
        show=False,
        format="svg"
    )
    print("  ✓ Realistic ML example (PNG + SVG)")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Neural Network Custom Neuron Labels Test Suite")
    print("=" * 60)
    
    test_basic_text_labels()
    test_latex_math_labels()
    test_complex_latex()
    test_left_and_right_positioning()
    test_mixed_labels()
    test_show_hide_control()
    test_with_neuron_indices()
    test_with_collapsed_layers()
    test_font_sizes()
    test_realistic_example()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("Check the 'test_outputs/neuron_labels' directory for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
