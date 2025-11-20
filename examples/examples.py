"""
Example Usage of NN_PLOT_MODULE

This file demonstrates how to use the NeuralNetwork class to create and manage
neural network structures with parent-child relationships between layers.
"""

from NN_DEFINITION_UTILITIES import NeuralNetwork, LayerInfo, NetworkType


# Example 1: Linear (Sequential) Network
# =======================================
# When you don't specify parent_ids, layers are automatically connected
# in sequence, creating a linear structure.

nn_linear = NeuralNetwork(
    name="Simple Classifier",
    network_type=NetworkType.FEEDFORWARD,
    description="A basic feedforward neural network"
)

# Add layers sequentially - each layer automatically connects to the previous one
input_layer_id = nn_linear.add_layer(
    LayerInfo(num_neurons=784, name="Input Layer")
)

hidden1_id = nn_linear.add_layer(
    LayerInfo(num_neurons=128, activation="relu", name="Hidden Layer 1")
)

hidden2_id = nn_linear.add_layer(
    LayerInfo(num_neurons=64, activation="relu", name="Hidden Layer 2")
)

output_id = nn_linear.add_layer(
    LayerInfo(num_neurons=10, activation="softmax", name="Output Layer")
)

print(nn_linear)
print("\n" + "="*80 + "\n")


# Example 2: Branching Network
# ==============================
# You can create more complex architectures by specifying parent_ids

nn_branch = NeuralNetwork(
    name="Multi-Path Network",
    network_type=NetworkType.FEEDFORWARD,
    description="A network with parallel processing paths"
)

# Input layer (no parents)
input_id = nn_branch.add_layer(
    LayerInfo(num_neurons=100, name="Input")
)

# Shared hidden layer
shared_hidden = nn_branch.add_layer(
    LayerInfo(num_neurons=64, activation="relu", name="Shared Hidden")
)

# Create two parallel branches from the shared hidden layer
path1 = nn_branch.add_layer(
    LayerInfo(num_neurons=32, activation="relu", name="Path 1"),
    parent_ids=[shared_hidden]  # Explicitly connect to shared_hidden
)

path2 = nn_branch.add_layer(
    LayerInfo(num_neurons=32, activation="relu", name="Path 2"),
    parent_ids=[shared_hidden]  # Also connects to shared_hidden
)

# Merge both paths into the output
output = nn_branch.add_layer(
    LayerInfo(num_neurons=10, activation="softmax", name="Output"),
    parent_ids=[path1, path2]  # Multiple parents!
)

print(nn_branch)
print("\n" + "="*80 + "\n")


# Example 3: Querying Network Structure
# ======================================

print("Network Analysis:")
print(f"Is linear: {nn_branch.is_linear()}")
print(f"Total layers: {nn_branch.num_layers()}")
print(f"Total neurons: {nn_branch.get_total_neurons()}")

# Find root and leaf layers
root_layers = nn_branch.get_root_layers()
print(f"\nRoot layers (inputs): {[nn_branch.get_layer(lid).name for lid in root_layers]}")

leaf_layers = nn_branch.get_leaf_layers()
print(f"Leaf layers (outputs): {[nn_branch.get_layer(lid).name for lid in leaf_layers]}")

# Examine connections
print(f"\nParents of Output: {[nn_branch.get_layer(lid).name for lid in nn_branch.get_parents(output)]}")
print(f"Children of Shared Hidden: {[nn_branch.get_layer(lid).name for lid in nn_branch.get_children(shared_hidden)]}")

# Access layer by name
layer = nn_branch.get_layer_by_name("Path 1")
if layer:
    print(f"\nFound layer: {layer.name} with {layer.num_neurons} neurons")


# Example 4: Layer Management
# ============================

nn_manage = NeuralNetwork(name="Managed Network")

# Add layers
l1 = nn_manage.add_layer(LayerInfo(num_neurons=50, name="Layer 1"))
l2 = nn_manage.add_layer(LayerInfo(num_neurons=30, name="Layer 2"))
l3 = nn_manage.add_layer(LayerInfo(num_neurons=10, name="Layer 3"))

print("\n" + "="*80 + "\n")
print("Before removal:")
print(nn_manage)

# Remove a layer
nn_manage.remove_layer(l2)

print("\nAfter removing Layer 2:")
print(nn_manage)
