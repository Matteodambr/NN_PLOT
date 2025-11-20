"""
Test script to demonstrate the NeuralNetwork class functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, LayerInfo, NetworkType


def test_linear_network():
    """Test a simple linear (sequential) network."""
    print("=" * 60)
    print("Testing Linear Network")
    print("=" * 60)
    
    # Create a network
    nn = NeuralNetwork(
        name="Linear Classifier",
        network_type=NetworkType.FEEDFORWARD,
        description="A simple feedforward neural network with linear structure"
    )
    
    # Add layers sequentially (no parent specified, so they connect linearly)
    input_id = nn.add_layer(LayerInfo(num_neurons=784, name="Input"))
    hidden1_id = nn.add_layer(LayerInfo(num_neurons=256, activation="relu", name="Hidden1"))
    hidden2_id = nn.add_layer(LayerInfo(num_neurons=128, activation="relu", name="Hidden2"))
    output_id = nn.add_layer(LayerInfo(num_neurons=10, activation="softmax", name="Output"))
    
    # Display the network
    print(nn)
    print("\n" + repr(nn))
    print()


def test_branching_network():
    """Test a non-linear network with branching."""
    print("=" * 60)
    print("Testing Branching Network")
    print("=" * 60)
    
    # Create a network
    nn = NeuralNetwork(
        name="Multi-Branch Network",
        network_type=NetworkType.FEEDFORWARD,
        description="A network with branching structure"
    )
    
    # Add input layer
    input_id = nn.add_layer(LayerInfo(num_neurons=100, name="Input"))
    
    # Add first hidden layer
    hidden1_id = nn.add_layer(LayerInfo(num_neurons=64, activation="relu", name="Hidden1"))
    
    # Branch out to two parallel layers (both connected to Hidden1)
    branch1_id = nn.add_layer(
        LayerInfo(num_neurons=32, activation="relu", name="Branch1"),
        parent_ids=[hidden1_id]
    )
    branch2_id = nn.add_layer(
        LayerInfo(num_neurons=32, activation="relu", name="Branch2"),
        parent_ids=[hidden1_id]
    )
    
    # Merge branches into output layer (connected to both branches)
    output_id = nn.add_layer(
        LayerInfo(num_neurons=10, activation="softmax", name="Output"),
        parent_ids=[branch1_id, branch2_id]
    )
    
    # Display the network
    print(nn)
    print("\n" + repr(nn))
    print()
    
    # Test querying methods
    print("\nQuerying network structure:")
    print(f"Root layers: {[nn.get_layer(lid).name for lid in nn.get_root_layers()]}")
    print(f"Leaf layers: {[nn.get_layer(lid).name for lid in nn.get_leaf_layers()]}")
    print(f"Parents of Output: {[nn.get_layer(lid).name for lid in nn.get_parents(output_id)]}")
    print(f"Children of Hidden1: {[nn.get_layer(lid).name for lid in nn.get_children(hidden1_id)]}")
    print()


def test_layer_operations():
    """Test layer manipulation operations."""
    print("=" * 60)
    print("Testing Layer Operations")
    print("=" * 60)
    
    nn = NeuralNetwork(name="Test Network")
    
    # Add layers
    l1_id = nn.add_layer(LayerInfo(num_neurons=10, name="Layer1"))
    l2_id = nn.add_layer(LayerInfo(num_neurons=20, name="Layer2"))
    l3_id = nn.add_layer(LayerInfo(num_neurons=30, name="Layer3"))
    
    print("After adding 3 layers:")
    print(nn)
    print()
    
    # Get layer by name
    layer = nn.get_layer_by_name("Layer2")
    print(f"Found layer by name: {layer.name} with {layer.num_neurons} neurons")
    
    # Get layer ID by name
    layer_id = nn.get_layer_id_by_name("Layer2")
    print(f"Layer2 ID: {layer_id}")
    print()
    
    # Remove middle layer
    print("Removing Layer2...")
    nn.remove_layer(l2_id)
    print(nn)
    print()


if __name__ == "__main__":
    test_linear_network()
    test_branching_network()
    test_layer_operations()
