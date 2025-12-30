"""
Comprehensive test for branching network layouts
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer, VectorInput
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig

output_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("Comprehensive Branching Network Tests")
print("=" * 70)

# ==============================================================================
# Test 1: Simple two-head output
# ==============================================================================
print("\n[Test 1] Simple two-head output")
nn1 = NeuralNetwork("Simple Branch")
input1 = VectorInput(num_features=5, name="Input")
nn1.add_layer(input1)
hidden1 = FullyConnectedLayer(10, activation="relu", name="Hidden")
nn1.add_layer(hidden1, parent_ids=[input1.layer_id])
out1a = FullyConnectedLayer(3, activation="softmax", name="Out1")
nn1.add_layer(out1a, parent_ids=[hidden1.layer_id])
out1b = FullyConnectedLayer(3, activation="softmax", name="Out2")
nn1.add_layer(out1b, parent_ids=[hidden1.layer_id])

config1 = PlotConfig(show_layer_names=True, show_neuron_labels=True, background_color='white')
plot_network(nn1, config=config1, title="Test 1: Simple Two-Head",
             save_path=os.path.join(output_dir, "branch_test1_simple.png"), show=False)
print("✓ Created: branch_test1_simple.png")

# ==============================================================================
# Test 2: Three output heads
# ==============================================================================
print("\n[Test 2] Three output heads")
nn2 = NeuralNetwork("Three Heads")
input2 = VectorInput(num_features=5, name="Input")
nn2.add_layer(input2)
hidden2 = FullyConnectedLayer(10, activation="relu", name="Hidden")
nn2.add_layer(hidden2, parent_ids=[input2.layer_id])
out2a = FullyConnectedLayer(3, activation="softmax", name="Out1")
nn2.add_layer(out2a, parent_ids=[hidden2.layer_id])
out2b = FullyConnectedLayer(3, activation="softmax", name="Out2")
nn2.add_layer(out2b, parent_ids=[hidden2.layer_id])
out2c = FullyConnectedLayer(3, activation="softmax", name="Out3")
nn2.add_layer(out2c, parent_ids=[hidden2.layer_id])

config2 = PlotConfig(show_layer_names=True, show_neuron_labels=True, background_color='white')
plot_network(nn2, config=config2, title="Test 2: Three Output Heads",
             save_path=os.path.join(output_dir, "branch_test2_three_heads.png"), show=False)
print("✓ Created: branch_test2_three_heads.png")

# ==============================================================================
# Test 3: Different sized output heads
# ==============================================================================
print("\n[Test 3] Different sized output heads")
nn3 = NeuralNetwork("Different Sizes")
input3 = VectorInput(num_features=5, name="Input")
nn3.add_layer(input3)
hidden3 = FullyConnectedLayer(10, activation="relu", name="Hidden")
nn3.add_layer(hidden3, parent_ids=[input3.layer_id])
out3a = FullyConnectedLayer(15, activation="softmax", name="Large")
nn3.add_layer(out3a, parent_ids=[hidden3.layer_id])
out3b = FullyConnectedLayer(3, activation="softmax", name="Small")
nn3.add_layer(out3b, parent_ids=[hidden3.layer_id])

config3 = PlotConfig(show_layer_names=True, show_neuron_labels=True, background_color='white')
plot_network(nn3, config=config3, title="Test 3: Different Sized Outputs",
             save_path=os.path.join(output_dir, "branch_test3_different_sizes.png"), show=False)
print("✓ Created: branch_test3_different_sizes.png")

# ==============================================================================
# Test 4: CEAS2025-like network (collapsed layers)
# ==============================================================================
print("\n[Test 4] Large layers with collapsing (like CEAS2025)")
nn4 = NeuralNetwork("CEAS-like")
input4 = VectorInput(num_features=6, name="Input")
nn4.add_layer(input4)
h4_1 = FullyConnectedLayer(300, activation="relu", name="H1")
nn4.add_layer(h4_1, parent_ids=[input4.layer_id])
h4_2 = FullyConnectedLayer(300, activation="relu", name="H2")
nn4.add_layer(h4_2, parent_ids=[h4_1.layer_id])
h4_3 = FullyConnectedLayer(300, activation="relu", name="H3")
nn4.add_layer(h4_3, parent_ids=[h4_2.layer_id])
out4a = FullyConnectedLayer(7, activation="softmax", name="Head1")
nn4.add_layer(out4a, parent_ids=[h4_3.layer_id])
out4b = FullyConnectedLayer(7, activation="softmax", name="Head2")
nn4.add_layer(out4b, parent_ids=[h4_3.layer_id])

config4 = PlotConfig(
    show_layer_names=False,
    show_neuron_labels=False,
    background_color='white',
    max_neurons_per_layer=20,
    figsize=(14, 10)
)
plot_network(nn4, config=config4, title="Test 4: CEAS-like Network",
             save_path=os.path.join(output_dir, "branch_test4_ceas_like.png"), show=False)
print("✓ Created: branch_test4_ceas_like.png")

print("\n" + "=" * 70)
print("All branching tests completed successfully!")
print("=" * 70)
print(f"\nAll test images saved to: {output_dir}/")
