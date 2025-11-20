"""
Test script demonstrating the collapsed/squashed neuron feature for large layers.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig

print("=" * 60)
print("Testing Collapsed Layers Feature")
print("=" * 60)

# Test 1: Network with a very large layer
print("\n[Test 1] Network with large hidden layer (100 neurons)")
nn_large = NeuralNetwork("Large Layer Network")
nn_large.add_layer(FullyConnectedLayer(num_neurons=5, name="Input"))
nn_large.add_layer(FullyConnectedLayer(num_neurons=100, activation="relu", name="LargeHidden"))
nn_large.add_layer(FullyConnectedLayer(num_neurons=3, activation="softmax", name="Output"))

# Default: max 20 neurons shown, so large layer will be collapsed
config_default = PlotConfig(
    figsize=(14, 8),
    max_neurons_per_layer=20  # Default value
)

plot_network(
    nn_large,
    title="Network with Large Hidden Layer (100 neurons, max_display=20)",
    save_path=os.path.join(os.path.dirname(__file__), "test_outputs", "test_collapsed_default.png"),
    show=False,
    config=config_default
)
print("✓ Saved: test_collapsed_default.png")
print("  - Large layer (100 neurons) collapsed to show first 10, dots, last 9")

# Test 2: More aggressive collapsing
print("\n[Test 2] Very aggressive collapsing (max 10 neurons)")
config_aggressive = PlotConfig(
    figsize=(14, 8),
    max_neurons_per_layer=10  # More aggressive
)

plot_network(
    nn_large,
    title="Network with Aggressive Collapsing (max_display=10)",
    save_path=os.path.join(os.path.dirname(__file__), "test_outputs", "test_collapsed_aggressive.png"),
    show=False,
    config=config_aggressive
)
print("✓ Saved: test_collapsed_aggressive.png")
print("  - Large layer collapsed to show first 5, dots, last 4")

# Test 3: Network where all layers are large
print("\n[Test 3] All layers are large")
nn_all_large = NeuralNetwork("All Large Layers")
nn_all_large.add_layer(FullyConnectedLayer(num_neurons=50, name="Input"))
nn_all_large.add_layer(FullyConnectedLayer(num_neurons=100, activation="relu", name="Hidden1"))
nn_all_large.add_layer(FullyConnectedLayer(num_neurons=80, activation="relu", name="Hidden2"))
nn_all_large.add_layer(FullyConnectedLayer(num_neurons=30, activation="softmax", name="Output"))

config_all_large = PlotConfig(
    figsize=(16, 8),
    max_neurons_per_layer=15
)

plot_network(
    nn_all_large,
    title="All Layers Collapsed (all have >15 neurons)",
    save_path=os.path.join(os.path.dirname(__file__), "test_outputs", "test_all_collapsed.png"),
    show=False,
    config=config_all_large
)
print("✓ Saved: test_all_collapsed.png")
print("  - All layers collapsed since all exceed 15 neurons")

# Test 4: Mixed - some collapsed, some not
print("\n[Test 4] Mixed network (some layers collapsed, some not)")
nn_mixed = NeuralNetwork("Mixed Network")
nn_mixed.add_layer(FullyConnectedLayer(num_neurons=3, name="SmallInput"))
nn_mixed.add_layer(FullyConnectedLayer(num_neurons=50, activation="relu", name="LargeHidden"))
nn_mixed.add_layer(FullyConnectedLayer(num_neurons=8, activation="relu", name="MediumHidden"))
nn_mixed.add_layer(FullyConnectedLayer(num_neurons=2, activation="softmax", name="SmallOutput"))

config_mixed = PlotConfig(
    figsize=(14, 8),
    max_neurons_per_layer=15
)

plot_network(
    nn_mixed,
    title="Mixed Network (only middle layer collapsed)",
    save_path=os.path.join(os.path.dirname(__file__), "test_outputs", "test_mixed_collapse.png"),
    show=False,
    config=config_mixed
)
print("✓ Saved: test_mixed_collapse.png")
print("  - Only the 50-neuron layer is collapsed")

# Test 5: With neuron labels enabled
print("\n[Test 5] Collapsed layer with neuron indices shown")
nn_labeled = NeuralNetwork("Labeled Network")
nn_labeled.add_layer(FullyConnectedLayer(num_neurons=4, name="Input"))
nn_labeled.add_layer(FullyConnectedLayer(num_neurons=30, activation="relu", name="Hidden"))
nn_labeled.add_layer(FullyConnectedLayer(num_neurons=2, activation="softmax", name="Output"))

config_labeled = PlotConfig(
    figsize=(14, 8),
    max_neurons_per_layer=12,
    show_neuron_labels=True  # Show indices
)

plot_network(
    nn_labeled,
    title="Collapsed Layer with Neuron Indices (0, 1, ..., 28, 29)",
    save_path=os.path.join(os.path.dirname(__file__), "test_outputs", "test_collapsed_with_labels.png"),
    show=False,
    config=config_labeled
)
print("✓ Saved: test_collapsed_with_labels.png")
print("  - Labels show actual neuron indices (first few, last few)")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
print("\nFeature Summary:")
print("  • When a layer exceeds max_neurons_per_layer:")
print("    - First N neurons are shown")
print("    - Three vertical dots (...) indicate omitted neurons")
print("    - Last M neurons are shown")
print("  • Connections skip the dots position")
print("  • Neuron labels show actual indices (not display indices)")
print("  • Default max_neurons_per_layer = 20")
print("\nGenerated files:")
print("  1. test_collapsed_default.png - Default collapsing (max 20)")
print("  2. test_collapsed_aggressive.png - Aggressive collapsing (max 10)")
print("  3. test_all_collapsed.png - All layers collapsed")
print("  4. test_mixed_collapse.png - Mixed (some collapsed, some not)")
print("  5. test_collapsed_with_labels.png - With neuron indices")
