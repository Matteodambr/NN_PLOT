"""
Test script to demonstrate and validate the new output layer types.
Shows VectorOutput and GenericOutput in various scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.NN_DEFINITION_UTILITIES import (
    NeuralNetwork, VectorInput, FullyConnectedLayer, 
    VectorOutput, GenericOutput
)
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig, LayerStyle

# Create output directory
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
output_dir = os.path.join(project_root, "test_outputs")
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("Testing Output Layer Types")
print("=" * 60)

# Test 1: VectorOutput for classification
print("\n1. Testing VectorOutput (Classification)")
nn1 = NeuralNetwork("Classification Network")
nn1.add_layer(VectorInput(num_features=10, name="Input"))
nn1.add_layer(FullyConnectedLayer(num_neurons=20, activation="relu", name="Hidden 1"))
nn1.add_layer(FullyConnectedLayer(num_neurons=10, activation="relu", name="Hidden 2"))
nn1.add_layer(VectorOutput(
    num_neurons=5, 
    activation="softmax", 
    name="Output",
    neuron_labels=[r"Class $1$", r"Class $2$", r"Class $3$", r"Class $4$", r"Class $5$"],
    label_position="right"
))

print(nn1)
print(f"Has output layer: {nn1.has_output_layer()}")
print(f"Output layers: {[nn1.get_layer(lid).name for lid in nn1.get_output_layers()]}")

try:
    plot_network(
        nn1,
        title="Classification with VectorOutput",
        config=PlotConfig(
            show_neuron_text_labels=True,
            layer_names_show_activation=True
        ),
        save_path=os.path.join(output_dir, "output_layer_classification.png"),
        show=False
    )
    print("✓ VectorOutput plot created successfully")
except Exception as e:
    print(f"✗ VectorOutput plot failed: {e}")

# Test 2: GenericOutput for regression
print("\n2. Testing GenericOutput (Regression)")
nn2 = NeuralNetwork("Regression Network")
nn2.add_layer(VectorInput(num_features=15, name="Features"))
nn2.add_layer(FullyConnectedLayer(num_neurons=32, activation="relu", name="Hidden 1"))
nn2.add_layer(FullyConnectedLayer(num_neurons=16, activation="relu", name="Hidden 2"))
nn2.add_layer(GenericOutput(
    output_size=1,
    text="Regression",
    name="Output"
))

print(nn2)

try:
    plot_network(
        nn2,
        title="Regression with GenericOutput",
        config=PlotConfig(
            layer_names_show_activation=True
        ),
        save_path=os.path.join(output_dir, "output_layer_regression.png"),
        show=False
    )
    print("✓ GenericOutput plot created successfully")
except Exception as e:
    print(f"✗ GenericOutput plot failed: {e}")

# Test 3: Multi-output network with both types
print("\n3. Testing Multi-Output Network")
nn3 = NeuralNetwork("Multi-Output Network")
input_id = nn3.add_layer(VectorInput(num_features=8, name="Input"))
shared_id = nn3.add_layer(FullyConnectedLayer(num_neurons=16, activation="relu", name="Shared"))

# Branch 1: Classification with VectorOutput
class_id = nn3.add_layer(
    VectorOutput(num_neurons=3, activation="softmax", name="Classification"),
    parent_ids=[shared_id]
)

# Branch 2: Regression with GenericOutput
regr_id = nn3.add_layer(
    GenericOutput(output_size=1, text="Regression", name="Regression"),
    parent_ids=[shared_id]
)

print(nn3)
print(f"Output layers: {[nn3.get_layer(lid).name for lid in nn3.get_output_layers()]}")

try:
    plot_network(
        nn3,
        title="Multi-Output Network (Classification + Regression)",
        config=PlotConfig(
            layer_names_show_activation=True,
            branch_spacing=4.0
        ),
        save_path=os.path.join(output_dir, "output_layer_multi_output.png"),
        show=False
    )
    print("✓ Multi-output network plot created successfully")
except Exception as e:
    print(f"✗ Multi-output network plot failed: {e}")

# Test 4: GenericOutput with custom text
print("\n4. Testing GenericOutput with Custom Text")
nn4 = NeuralNetwork("Custom Output Text")
nn4.add_layer(VectorInput(num_features=5, name="Input"))
nn4.add_layer(FullyConnectedLayer(num_neurons=10, activation="relu", name="Hidden"))
nn4.add_layer(GenericOutput(
    output_size=3,
    text="Custom\nOutput",
    name="Output"
))

try:
    plot_network(
        nn4,
        title="GenericOutput with Custom Text",
        save_path=os.path.join(output_dir, "output_layer_custom_text.png"),
        show=False
    )
    print("✓ Custom text GenericOutput plot created successfully")
except Exception as e:
    print(f"✗ Custom text GenericOutput plot failed: {e}")

# Test 5: Styled output layers
print("\n5. Testing Styled Output Layers")
nn5 = NeuralNetwork("Styled Outputs")
nn5.add_layer(VectorInput(num_features=6, name="Input"))
nn5.add_layer(FullyConnectedLayer(num_neurons=8, activation="relu", name="Hidden"))
output_id = nn5.add_layer(VectorOutput(num_neurons=4, activation="softmax", name="Output"))

# Style the output layer
layer_styles = {
    "Output": LayerStyle(
        neuron_fill_color='lightcoral',
        neuron_edge_color='darkred',
        neuron_edge_width=2.0,
        box_around_layer=True,
        box_fill_color='lightyellow',
        box_edge_color='orange',
        box_edge_width=2.0
    )
}

try:
    plot_network(
        nn5,
        title="Styled VectorOutput Layer",
        config=PlotConfig(
            layer_styles=layer_styles
        ),
        save_path=os.path.join(output_dir, "output_layer_styled.png"),
        show=False
    )
    print("✓ Styled output layer plot created successfully")
except Exception as e:
    print(f"✗ Styled output layer plot failed: {e}")

print("\n" + "=" * 60)
print("All output layer tests completed!")
print("Check the test_outputs/ directory for generated plots.")
print("=" * 60)
