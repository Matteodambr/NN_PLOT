"""
Example Usage of NN_PLOTTING_UTILITIES with Specific Layer Types

This file demonstrates how to use the updated module with the new layer type system.
Currently supports: FullyConnectedLayer (Dense layers)
"""

from NN_DEFINITION_UTILITIES import (
    NeuralNetwork,
    FullyConnectedLayer,
    NetworkType,
    LayerType
)


def example_1_simple_classifier():
    """Example 1: Simple feedforward classifier with fully connected layers."""
    print("=" * 80)
    print("Example 1: Simple Feedforward Classifier")
    print("=" * 80)
    
    # Create a neural network
    nn = NeuralNetwork(
        name="MNIST Classifier",
        network_type=NetworkType.FEEDFORWARD,
        description="A simple classifier for MNIST digits"
    )
    
    # Add fully connected layers sequentially
    # Input layer (784 features for 28x28 images)
    nn.add_layer(FullyConnectedLayer(
        num_neurons=784,
        name="Input"
    ))
    
    # Hidden layer 1 with ReLU activation
    nn.add_layer(FullyConnectedLayer(
        num_neurons=256,
        activation="relu",
        name="Hidden 1"
    ))
    
    # Hidden layer 2 with ReLU activation
    nn.add_layer(FullyConnectedLayer(
        num_neurons=128,
        activation="relu",
        name="Hidden 2"
    ))
    
    # Output layer with softmax for 10 classes
    nn.add_layer(FullyConnectedLayer(
        num_neurons=10,
        activation="softmax",
        name="Output"
    ))
    
    print(nn)
    print("\n" + "=" * 80 + "\n")


def example_2_deep_network():
    """Example 2: Deeper network with more layers."""
    print("=" * 80)
    print("Example 2: Deep Neural Network")
    print("=" * 80)
    
    nn = NeuralNetwork(
        name="Deep Classifier",
        network_type=NetworkType.FEEDFORWARD,
        description="A deeper network with 6 hidden layers"
    )
    
    # Input
    nn.add_layer(FullyConnectedLayer(num_neurons=512, name="Input"))
    
    # Deep hidden layers with decreasing size
    for i, size in enumerate([256, 128, 64, 32, 16], 1):
        nn.add_layer(FullyConnectedLayer(
            num_neurons=size,
            activation="relu",
            name=f"Hidden {i}"
        ))
    
    # Output
    nn.add_layer(FullyConnectedLayer(
        num_neurons=3,
        activation="softmax",
        name="Output"
    ))
    
    print(nn)
    print(f"\nNetwork depth: {nn.num_layers()} layers")
    print(f"Is linear: {nn.is_linear()}")
    print("\n" + "=" * 80 + "\n")


def example_3_branching_architecture():
    """Example 3: Non-linear architecture with branches."""
    print("=" * 80)
    print("Example 3: Multi-Branch Architecture")
    print("=" * 80)
    
    nn = NeuralNetwork(
        name="Multi-Task Network",
        network_type=NetworkType.FEEDFORWARD,
        description="Network with shared backbone and task-specific heads"
    )
    
    # Shared backbone
    input_id = nn.add_layer(FullyConnectedLayer(
        num_neurons=200,
        name="Input Features"
    ))
    
    shared1_id = nn.add_layer(FullyConnectedLayer(
        num_neurons=128,
        activation="relu",
        name="Shared Layer 1"
    ))
    
    shared2_id = nn.add_layer(FullyConnectedLayer(
        num_neurons=64,
        activation="relu",
        name="Shared Layer 2"
    ))
    
    # Task 1 branch (Classification)
    task1_hidden = nn.add_layer(
        FullyConnectedLayer(
            num_neurons=32,
            activation="relu",
            name="Task 1 Hidden"
        ),
        parent_ids=[shared2_id]
    )
    
    task1_output = nn.add_layer(
        FullyConnectedLayer(
            num_neurons=5,
            activation="softmax",
            name="Task 1 Output (Classification)"
        ),
        parent_ids=[task1_hidden]
    )
    
    # Task 2 branch (Regression)
    task2_hidden = nn.add_layer(
        FullyConnectedLayer(
            num_neurons=32,
            activation="relu",
            name="Task 2 Hidden"
        ),
        parent_ids=[shared2_id]
    )
    
    task2_output = nn.add_layer(
        FullyConnectedLayer(
            num_neurons=1,
            activation="linear",
            name="Task 2 Output (Regression)"
        ),
        parent_ids=[task2_hidden]
    )
    
    print(nn)
    print(f"\nRoot layers: {[nn.get_layer(lid).name for lid in nn.get_root_layers()]}")
    print(f"Leaf layers: {[nn.get_layer(lid).name for lid in nn.get_leaf_layers()]}")
    print("\n" + "=" * 80 + "\n")


def example_4_layer_properties():
    """Example 4: Working with layer properties."""
    print("=" * 80)
    print("Example 4: Layer Properties and Queries")
    print("=" * 80)
    
    nn = NeuralNetwork(name="Property Demo")
    
    # Create layers with different properties
    layer1 = FullyConnectedLayer(
        num_neurons=100,
        activation="relu",
        name="Feature Extractor",
        use_bias=True
    )
    
    layer2 = FullyConnectedLayer(
        num_neurons=50,
        activation="tanh",
        name="Encoder",
        use_bias=False  # No bias
    )
    
    layer3 = FullyConnectedLayer(
        num_neurons=10,
        activation="softmax",
        name="Classifier"
    )
    
    # Add to network
    id1 = nn.add_layer(layer1)
    id2 = nn.add_layer(layer2)
    id3 = nn.add_layer(layer3)
    
    print("Network structure:")
    print(nn)
    
    # Query layers
    print("\nLayer queries:")
    encoder = nn.get_layer_by_name("Encoder")
    if encoder:
        print(f"\nFound 'Encoder' layer:")
        print(f"  Type: {encoder.layer_type.value}")
        print(f"  Neurons: {encoder.num_neurons}")
        print(f"  Activation: {encoder.activation}")
        print(f"  Use bias: {encoder.use_bias}")
        print(f"  Output size: {encoder.get_output_size()}")
    
    # Get parents and children
    print(f"\nConnections for 'Encoder':")
    encoder_id = nn.get_layer_id_by_name("Encoder")
    if encoder_id:
        parents = nn.get_parents(encoder_id)
        children = nn.get_children(encoder_id)
        print(f"  Parents: {[nn.get_layer(p).name for p in parents]}")
        print(f"  Children: {[nn.get_layer(c).name for c in children]}")
    
    print("\n" + "=" * 80 + "\n")


def example_5_custom_architectures():
    """Example 5: Creating custom architectures."""
    print("=" * 80)
    print("Example 5: Custom Architecture - Autoencoder")
    print("=" * 80)
    
    nn = NeuralNetwork(
        name="Autoencoder",
        network_type=NetworkType.FEEDFORWARD,
        description="Symmetric autoencoder for dimensionality reduction"
    )
    
    # Encoder (compressing)
    nn.add_layer(FullyConnectedLayer(num_neurons=784, name="Input"))
    nn.add_layer(FullyConnectedLayer(num_neurons=512, activation="relu", name="Encoder 1"))
    nn.add_layer(FullyConnectedLayer(num_neurons=256, activation="relu", name="Encoder 2"))
    nn.add_layer(FullyConnectedLayer(num_neurons=128, activation="relu", name="Encoder 3"))
    
    # Bottleneck (latent representation)
    nn.add_layer(FullyConnectedLayer(num_neurons=32, activation="relu", name="Bottleneck"))
    
    # Decoder (reconstructing)
    nn.add_layer(FullyConnectedLayer(num_neurons=128, activation="relu", name="Decoder 1"))
    nn.add_layer(FullyConnectedLayer(num_neurons=256, activation="relu", name="Decoder 2"))
    nn.add_layer(FullyConnectedLayer(num_neurons=512, activation="relu", name="Decoder 3"))
    nn.add_layer(FullyConnectedLayer(num_neurons=784, activation="sigmoid", name="Output"))
    
    print(nn)
    print(f"\nBottleneck dimension: {nn.get_layer_by_name('Bottleneck').get_output_size()}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    example_1_simple_classifier()
    example_2_deep_network()
    example_3_branching_architecture()
    example_4_layer_properties()
    example_5_custom_architectures()
    
    print("All examples completed successfully!")
