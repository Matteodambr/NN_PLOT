"""
Quick demo showing custom neuron labels with LaTeX support.
Creates a simple but comprehensive example.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer
from NN_PLOTTING_UTILITIES import plot_network, PlotConfig

os.makedirs("test_outputs", exist_ok=True)

# Example 1: Realistic machine learning model with descriptive labels
print("Creating labeled neural network example...")

nn = NeuralNetwork("Customer Churn Prediction")

# Input layer - features with plain text labels on the left
nn.add_layer(FullyConnectedLayer(
    num_neurons=5,
    name="Input Features",
    neuron_labels=[
        "Monthly Charges",
        "Contract Length",
        "Support Tickets",
        "Usage Minutes",
        "Customer Age"
    ],
    label_position="left"
))

# Hidden layers - no labels needed
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

# Output layer - predictions with labels on the right
nn.add_layer(FullyConnectedLayer(
    num_neurons=2,
    activation="softmax",
    name="Churn Prediction",
    neuron_labels=["Will Stay", "Will Churn"],
    label_position="right"
))

# Create plot with labels enabled
config = PlotConfig(
    show_neuron_text_labels=True,
    neuron_text_label_fontsize=10,
    figsize=(14, 8)
)

plot_network(
    nn,
    config=config,
    title="Customer Churn Prediction Model",
    save_path="test_outputs/demo_labeled_network.png",
    show=False,
    dpi=300
)

print("✅ Created: test_outputs/demo_labeled_network.png")

# Example 2: LaTeX mathematical notation
print("Creating LaTeX math example...")

nn_math = NeuralNetwork("Mathematical Model")

# Input with LaTeX math notation
nn_math.add_layer(FullyConnectedLayer(
    num_neurons=3,
    name="Input",
    neuron_labels=[r"$x_1$", r"$x_2$", r"$x_3$"],
    label_position="left"
))

nn_math.add_layer(FullyConnectedLayer(
    num_neurons=5,
    activation="tanh",
    name="Hidden",
    neuron_labels=[r"$h_1$", r"$h_2$", r"$h_3$", r"$h_4$", r"$h_5$"],
    label_position="left"
))

# Output with predicted values
nn_math.add_layer(FullyConnectedLayer(
    num_neurons=2,
    activation="sigmoid",
    name="Output",
    neuron_labels=[r"$\hat{y}_1$", r"$\hat{y}_2$"],
    label_position="right"
))

config_math = PlotConfig(
    show_neuron_text_labels=True,
    neuron_text_label_fontsize=12
)

plot_network(
    nn_math,
    config=config_math,
    title="Neural Network with LaTeX Notation",
    save_path="test_outputs/demo_latex_labels.png",
    show=False,
    dpi=300
)

# Also save as SVG for presentations
plot_network(
    nn_math,
    config=config_math,
    title="Neural Network with LaTeX Notation",
    save_path="test_outputs/demo_latex_labels.svg",
    show=False,
    format="svg"
)

print("✅ Created: test_outputs/demo_latex_labels.png")
print("✅ Created: test_outputs/demo_latex_labels.svg (scalable vector)")

print("\n" + "="*60)
print("Demo complete! Check the 'test_outputs' folder for:")
print("  1. demo_labeled_network.png - Real-world example with text labels")
print("  2. demo_latex_labels.png - LaTeX mathematical notation")
print("  3. demo_latex_labels.svg - Scalable vector version")
print("="*60)
