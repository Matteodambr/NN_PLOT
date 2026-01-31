"""
Test branching network with one parent and multiple children
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer, VectorInput, VectorOutput
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig

# Create output directory
output_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("Testing Branching Network Issue")
print("=" * 70)

# Simple test: one parent with two children
nn_test = NeuralNetwork("Test Branching")

input_layer = VectorInput(num_features=6, name="Input")
nn_test.add_layer(input_layer)

hidden1 = FullyConnectedLayer(10, activation="relu", name="Hidden1")
nn_test.add_layer(hidden1, parent_ids=[input_layer.layer_id])

hidden2 = FullyConnectedLayer(10, activation="relu", name="Hidden2")
nn_test.add_layer(hidden2, parent_ids=[hidden1.layer_id])

hidden3 = FullyConnectedLayer(10, activation="relu", name="Hidden3")
nn_test.add_layer(hidden3, parent_ids=[hidden2.layer_id])

# Two output heads from the same parent
output1 = VectorOutput(7, activation="softmax", name="Output1")
nn_test.add_layer(output1, parent_ids=[hidden3.layer_id])

output2 = VectorOutput(7, activation="softmax", name="Output2")
nn_test.add_layer(output2, parent_ids=[hidden3.layer_id])

config = PlotConfig(
    figsize=(12, 8),
    show_layer_names=True,
    show_neuron_labels=True,
    background_color='white'
)

plot_network(
    nn_test,
    config=config,
    title="Test: One Parent, Two Children",
    save_path=os.path.join(output_dir, "test_branching.png"),
    show=False
)

print("✓ Created: test_branching.png")
print("\nNetwork structure:")
print(f"  Input (6) -> Hidden1 (10) -> Hidden2 (10) -> Hidden3 (10)")
print(f"                                                   ├─> Output1 (7)")
print(f"                                                   └─> Output2 (7)")
