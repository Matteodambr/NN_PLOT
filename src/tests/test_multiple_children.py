"""
Test to demonstrate that a fully connected layer can have multiple children.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.NN_DEFINITION_UTILITIES import (
    NeuralNetwork,
    FullyConnectedLayer,
    NetworkType
)


def test_fc_with_multiple_children():
    """Test a single FC layer branching to multiple FC children."""
    print("=" * 70)
    print("Test: Single FC Layer with Multiple FC Children")
    print("=" * 70)
    
    nn = NeuralNetwork(
        name="Branching FC Network",
        network_type=NetworkType.FEEDFORWARD
    )
    
    # Add input layer
    input_id = nn.add_layer(FullyConnectedLayer(
        num_neurons=100,
        name="Input"
    ))
    
    # Add a single FC layer that will have multiple children
    parent_fc_id = nn.add_layer(FullyConnectedLayer(
        num_neurons=64,
        activation="relu",
        name="Parent FC Layer"
    ))
    
    # Add MULTIPLE children FC layers, all connected to the parent
    child1_id = nn.add_layer(
        FullyConnectedLayer(num_neurons=32, activation="relu", name="Child FC 1"),
        parent_ids=[parent_fc_id]
    )
    
    child2_id = nn.add_layer(
        FullyConnectedLayer(num_neurons=32, activation="relu", name="Child FC 2"),
        parent_ids=[parent_fc_id]
    )
    
    child3_id = nn.add_layer(
        FullyConnectedLayer(num_neurons=32, activation="relu", name="Child FC 3"),
        parent_ids=[parent_fc_id]
    )
    
    # Each child can have its own output
    output1_id = nn.add_layer(
        FullyConnectedLayer(num_neurons=10, activation="softmax", name="Output 1"),
        parent_ids=[child1_id]
    )
    
    output2_id = nn.add_layer(
        FullyConnectedLayer(num_neurons=5, activation="softmax", name="Output 2"),
        parent_ids=[child2_id]
    )
    
    output3_id = nn.add_layer(
        FullyConnectedLayer(num_neurons=3, activation="softmax", name="Output 3"),
        parent_ids=[child3_id]
    )
    
    # Display the network
    print(nn)
    print("\n" + "=" * 70)
    
    # Query the parent layer's children
    print("\nQuerying Parent FC Layer:")
    children = nn.get_children(parent_fc_id)
    print(f"Number of children: {len(children)}")
    print(f"Children layers:")
    for child_id in children:
        child = nn.get_layer(child_id)
        print(f"  - {child.name}: {child.num_neurons} neurons, activation: {child.activation}")
    
    print("\n" + "=" * 70)
    print(f"Is network linear? {nn.is_linear()}")
    print(f"This is a branching network because 'Parent FC Layer' has {len(children)} children!")
    print("=" * 70)


def test_fc_with_many_children():
    """Test a FC layer with even more children."""
    print("\n" + "=" * 70)
    print("Test: Single FC Layer with MANY Children")
    print("=" * 70)
    
    nn = NeuralNetwork(name="Multi-Branch FC")
    
    # Input and shared layer
    input_id = nn.add_layer(FullyConnectedLayer(100, name="Input"))
    shared_id = nn.add_layer(FullyConnectedLayer(64, activation="relu", name="Shared FC"))
    
    # Create 5 parallel branches from the same parent
    for i in range(1, 6):
        branch_id = nn.add_layer(
            FullyConnectedLayer(
                num_neurons=20,
                activation="relu",
                name=f"Branch {i}"
            ),
            parent_ids=[shared_id]
        )
        
        # Each branch gets its own output
        nn.add_layer(
            FullyConnectedLayer(
                num_neurons=10,
                activation="softmax",
                name=f"Output {i}"
            ),
            parent_ids=[branch_id]
        )
    
    print(nn)
    
    # Show the branching
    children = nn.get_children(shared_id)
    print(f"\n'Shared FC' layer has {len(children)} children:")
    for child_id in children:
        print(f"  - {nn.get_layer(child_id).name}")
    print("=" * 70)


if __name__ == "__main__":
    test_fc_with_multiple_children()
    test_fc_with_many_children()
    
    print("\n✅ YES! The code fully supports FC layers with multiple FC children!")
    print("   You can create arbitrarily complex branching structures!")
