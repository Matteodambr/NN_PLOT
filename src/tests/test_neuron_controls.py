"""
Test script demonstrating neuron numbering control and custom collapse distribution.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig

print("=" * 70)
print("Testing Neuron Numbering and Collapse Distribution Controls")
print("=" * 70)

# Create a test network with a large layer
nn = NeuralNetwork("Control Test Network")
nn.add_layer(FullyConnectedLayer(num_neurons=5, name="Input"))
nn.add_layer(FullyConnectedLayer(num_neurons=50, activation="relu", name="LargeHidden"))
nn.add_layer(FullyConnectedLayer(num_neurons=3, activation="softmax", name="Output"))

# Test 1: Without neuron labels (default)
print("\n[Test 1] Without neuron numbering (show_neuron_labels=False)")
config_no_labels = PlotConfig(
    figsize=(14, 8),
    show_neuron_labels=False,  # No numbers on neurons
    max_neurons_per_layer=20,
    collapse_neurons_start=10,
    collapse_neurons_end=9
)

plot_network(
    nn,
    title="Network WITHOUT Neuron Numbers",
    save_path=os.path.join(os.path.dirname(__file__), "test_outputs", "test_no_neuron_numbers.png"),
    show=False,
    config=config_no_labels
)
print("✓ Saved: test_no_neuron_numbers.png")
print("  - Neurons displayed without index numbers")

# Test 2: With neuron labels
print("\n[Test 2] With neuron numbering (show_neuron_labels=True)")
config_with_labels = PlotConfig(
    figsize=(14, 8),
    show_neuron_labels=True,  # Show numbers on neurons
    max_neurons_per_layer=20,
    collapse_neurons_start=10,
    collapse_neurons_end=9
)

plot_network(
    nn,
    title="Network WITH Neuron Numbers (0, 1, ..., 48, 49)",
    save_path=os.path.join(os.path.dirname(__file__), "test_outputs", "test_with_neuron_numbers.png"),
    show=False,
    config=config_with_labels
)
print("✓ Saved: test_with_neuron_numbers.png")
print("  - Each neuron shows its index number")
print("  - Large layer shows: neurons 0-9, dots, neurons 41-49")

# Test 3: Custom collapse distribution - more at start
print("\n[Test 3] Custom collapse: More neurons at start (15 start, 4 end)")
config_more_start = PlotConfig(
    figsize=(14, 8),
    show_neuron_labels=True,
    max_neurons_per_layer=20,
    collapse_neurons_start=15,  # Show more at start
    collapse_neurons_end=4      # Show fewer at end
)

plot_network(
    nn,
    title="Custom Collapse: 15 start + dots + 4 end",
    save_path=os.path.join(os.path.dirname(__file__), "test_outputs", "test_collapse_more_start.png"),
    show=False,
    config=config_more_start
)
print("✓ Saved: test_collapse_more_start.png")
print("  - Large layer shows: neurons 0-14, dots, neurons 46-49")

# Test 4: Custom collapse distribution - more at end
print("\n[Test 4] Custom collapse: More neurons at end (5 start, 14 end)")
config_more_end = PlotConfig(
    figsize=(14, 8),
    show_neuron_labels=True,
    max_neurons_per_layer=20,
    collapse_neurons_start=5,   # Show fewer at start
    collapse_neurons_end=14     # Show more at end
)

plot_network(
    nn,
    title="Custom Collapse: 5 start + dots + 14 end",
    save_path=os.path.join(os.path.dirname(__file__), "test_outputs", "test_collapse_more_end.png"),
    show=False,
    config=config_more_end
)
print("✓ Saved: test_collapse_more_end.png")
print("  - Large layer shows: neurons 0-4, dots, neurons 36-49")

# Test 5: Minimal collapse (just a few neurons on each side)
print("\n[Test 5] Minimal collapse: Few neurons on each side (3 start, 3 end)")
config_minimal = PlotConfig(
    figsize=(14, 8),
    show_neuron_labels=True,
    max_neurons_per_layer=20,
    collapse_neurons_start=3,
    collapse_neurons_end=3
)

plot_network(
    nn,
    title="Minimal Collapse: 3 start + dots + 3 end",
    save_path=os.path.join(os.path.dirname(__file__), "test_outputs", "test_collapse_minimal.png"),
    show=False,
    config=config_minimal
)
print("✓ Saved: test_collapse_minimal.png")
print("  - Large layer shows: neurons 0-2, dots, neurons 47-49")

# Test 6: Symmetric collapse (equal on both sides)
print("\n[Test 6] Symmetric collapse: Equal on both sides (7 start, 7 end)")
config_symmetric = PlotConfig(
    figsize=(14, 8),
    show_neuron_labels=True,
    max_neurons_per_layer=20,
    collapse_neurons_start=7,
    collapse_neurons_end=7
)

plot_network(
    nn,
    title="Symmetric Collapse: 7 start + dots + 7 end",
    save_path=os.path.join(os.path.dirname(__file__), "test_outputs", "test_collapse_symmetric.png"),
    show=False,
    config=config_symmetric
)
print("✓ Saved: test_collapse_symmetric.png")
print("  - Large layer shows: neurons 0-6, dots, neurons 43-49")

print("\n" + "=" * 70)
print("All tests completed!")
print("=" * 70)

print("\n📊 Feature Summary:")
print("\n1. NEURON NUMBERING CONTROL:")
print("   • show_neuron_labels=False → No numbers on neurons (cleaner look)")
print("   • show_neuron_labels=True  → Show index on each neuron")
print("   • Collapsed layers show actual indices (not display positions)")

print("\n2. COLLAPSE DISTRIBUTION CONTROL:")
print("   • collapse_neurons_start: Number of neurons at the beginning")
print("   • collapse_neurons_end: Number of neurons at the end")
print("   • Example: start=10, end=9 means:")
print("     - Show neurons 0-9")
print("     - Show dots (...)")
print("     - Show last 9 neurons")

print("\n3. USE CASES:")
print("   • More at start: Good for input layers (see initial features)")
print("   • More at end: Good for output layers (see final outputs)")
print("   • Minimal: For very large layers (100+) where space is tight")
print("   • Symmetric: Balanced view of both ends")

print("\n📁 Generated files:")
print("   1. test_no_neuron_numbers.png - Without numbering")
print("   2. test_with_neuron_numbers.png - With numbering (10+9)")
print("   3. test_collapse_more_start.png - Asymmetric (15+4)")
print("   4. test_collapse_more_end.png - Asymmetric (5+14)")
print("   5. test_collapse_minimal.png - Minimal (3+3)")
print("   6. test_collapse_symmetric.png - Symmetric (7+7)")
