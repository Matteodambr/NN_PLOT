"""
Test script to verify the updated module with specific layer types.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.NN_DEFINITION_UTILITIES import (
    NeuralNetwork,
    FullyConnectedLayer,
    VectorInput,
    NetworkType,
    LayerType,
    VectorOutput
)


def test_fully_connected_layers():
    """Test creating a network with fully connected layers."""
    print("=" * 60)
    print("Testing Fully Connected Layer Type")
    print("=" * 60)
    
    # Create a network
    nn = NeuralNetwork(
        name="FC Classifier",
        network_type=NetworkType.FEEDFORWARD,
        description="A network with only fully connected layers"
    )
    
    # Add fully connected layers
    input_layer = FullyConnectedLayer(
        num_neurons=784,
        name="Input Layer"
    )
    input_id = nn.add_layer(input_layer)
    print(f"Added: {input_layer}")
    print(f"Layer ID: {input_id[:8]}...")
    print(f"Output size: {input_layer.get_output_size()}")
    print(f"Layer type: {input_layer.layer_type.value}")
    print()
    
    # Add hidden layer with activation
    hidden1 = FullyConnectedLayer(
        num_neurons=256,
        activation="relu",
        name="Hidden 1"
    )
    hidden1_id = nn.add_layer(hidden1)
    print(f"Added: {hidden1}")
    print()
    
    # Add another hidden layer
    hidden2 = FullyConnectedLayer(
        num_neurons=128,
        activation="relu",
        name="Hidden 2",
        use_bias=True
    )
    hidden2_id = nn.add_layer(hidden2)
    print(f"Added: {hidden2}")
    print()
    
    # Add output layer
    output = FullyConnectedLayer(
        num_neurons=10,
        activation="softmax",
        name="Output Layer"
    )
    output_id = nn.add_layer(output)
    print(f"Added: {output}")
    print()
    
    # Display the complete network
    print("\n" + "=" * 60)
    print("Complete Network Structure")
    print("=" * 60)
    print(nn)
    print("\n" + repr(nn))
    print()


def test_branching_with_fc_layers():
    """Test a branching network with fully connected layers."""
    print("=" * 60)
    print("Testing Branching Network with FC Layers")
    print("=" * 60)
    
    nn = NeuralNetwork(
        name="Multi-Branch FC Network",
        network_type=NetworkType.FEEDFORWARD
    )
    
    # Input layer
    input_id = nn.add_layer(
        VectorInput(num_features=100, name="Input")
    )
    
    # Shared hidden layer
    shared_id = nn.add_layer(
        FullyConnectedLayer(num_neurons=64, activation="relu", name="Shared")
    )
    
    # Create two branches
    branch1_id = nn.add_layer(
        FullyConnectedLayer(num_neurons=32, activation="relu", name="Branch 1"),
        parent_ids=[shared_id]
    )
    
    branch2_id = nn.add_layer(
        FullyConnectedLayer(num_neurons=32, activation="relu", name="Branch 2"),
        parent_ids=[shared_id]
    )
    
    # Merge branches
    output_id = nn.add_layer(
        VectorOutput(num_neurons=10, activation="softmax", name="Output"),
        parent_ids=[branch1_id, branch2_id]
    )
    
    print(nn)
    print()
    
    # Query specific layers
    print("Layer Queries:")
    output_layer = nn.get_layer(output_id)
    print(f"Output layer: {output_layer}")
    print(f"Output layer type: {output_layer.layer_type.value}")
    print(f"Output size: {output_layer.get_output_size()}")
    
    # Query by name
    branch1 = nn.get_layer_by_name("Branch 1")
    if branch1:
        print(f"\nFound by name: {branch1}")
        print(f"Neurons: {branch1.num_neurons}")
    print()


def test_layer_properties():
    """Test accessing layer properties."""
    print("=" * 60)
    print("Testing Layer Properties")
    print("=" * 60)
    
    # Create a fully connected layer
    fc_layer = FullyConnectedLayer(
        num_neurons=128,
        activation="relu",
        name="Test Layer",
        use_bias=True
    )
    
    print(f"Layer Type: {fc_layer.layer_type}")
    print(f"Layer Name: {fc_layer.name}")
    print(f"Number of Neurons: {fc_layer.num_neurons}")
    print(f"Activation: {fc_layer.activation}")
    print(f"Use Bias: {fc_layer.use_bias}")
    print(f"Output Size: {fc_layer.get_output_size()}")
    print(f"Layer ID: {fc_layer.layer_id}")
    print(f"\nString representation: {fc_layer}")
    print(f"Repr: {repr(fc_layer)}")
    print()


def test_error_handling():
    """Test error handling."""
    print("=" * 60)
    print("Testing Error Handling")
    print("=" * 60)
    
    # Test invalid neuron count
    try:
        invalid_layer = FullyConnectedLayer(num_neurons=0, name="Invalid")
        print("ERROR: Should have raised ValueError for 0 neurons")
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    # Test invalid layer type in add_layer
    try:
        nn = NeuralNetwork(name="Test")
        nn.add_layer("not a layer")  # Wrong type
        print("ERROR: Should have raised TypeError")
    except TypeError as e:
        print(f"✓ Correctly caught error: {e}")
    
    print()


if __name__ == "__main__":
    test_fully_connected_layers()
    test_branching_with_fc_layers()
    test_layer_properties()
    test_error_handling()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
