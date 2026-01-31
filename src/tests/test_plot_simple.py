"""
Simple test to verify plotting functionality works.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer, VectorInput, VectorOutput
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig

# Create output directory at project root
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
output_dir = os.path.join(project_root, "test_outputs")
os.makedirs(output_dir, exist_ok=True)

# Create a simple network
nn = NeuralNetwork(name="Test Network")
nn.add_layer(VectorInput(num_features=4, name="Input"))
nn.add_layer(FullyConnectedLayer(num_neurons=5, activation="relu", name="Hidden"))
nn.add_layer(VectorOutput(num_neurons=3, activation="softmax", name="Output"))

print("Network created:")
print(nn)
print("\nGenerating plot...")

# Plot without showing (just save)
plot_network(
    nn,
    title="Simple Test Network",
    save_path=os.path.join(output_dir, "test_plot.png"),
    show=False  # Don't show, just save
)

print("✓ Plot saved as 'test_plot.png'")
print("✓ Plotting functionality is working!")
