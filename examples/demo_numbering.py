"""
Quick demo showing the difference between normal and reversed neuron numbering.
Creates side-by-side comparison plots.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer, VectorInput
from NN_PLOTTING_UTILITIES import plot_network, PlotConfig

# Create a simple network
nn = NeuralNetwork("Numbering Demo")
nn.add_layer(VectorInput(num_features=6, name="Input"))
nn.add_layer(FullyConnectedLayer(4, activation="relu", name="Hidden"))
nn.add_layer(FullyConnectedLayer(3, activation="softmax", name="Output"))

os.makedirs("test_outputs", exist_ok=True)

# Normal numbering (0 at top)
print("Creating normal numbering example...")
config_normal = PlotConfig(
    show_neuron_labels=True,
    neuron_numbering_reversed=False
)
plot_network(
    nn,
    config=config_normal,
    title="Normal Numbering: 0 at Top → N-1 at Bottom",
    save_path="test_outputs/demo_normal_numbering.png",
    show=False,
    dpi=300
)

# Reversed numbering (N-1 at top)
print("Creating reversed numbering example...")
config_reversed = PlotConfig(
    show_neuron_labels=True,
    neuron_numbering_reversed=True
)
plot_network(
    nn,
    config=config_reversed,
    title="Reversed Numbering: N-1 at Top → 0 at Bottom",
    save_path="test_outputs/demo_reversed_numbering.png",
    show=False,
    dpi=300
)

# SVG version for scalability
print("Creating SVG version...")
plot_network(
    nn,
    config=config_reversed,
    title="Reversed Numbering (SVG - Infinitely Scalable)",
    save_path="test_outputs/demo_reversed_numbering.svg",
    show=False,
    format="svg"
)

print("\n✅ Demo complete!")
print("Check the 'test_outputs' folder for:")
print("  - demo_normal_numbering.png (PNG, 300 DPI)")
print("  - demo_reversed_numbering.png (PNG, 300 DPI)")
print("  - demo_reversed_numbering.svg (SVG, scalable)")
