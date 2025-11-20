"""
Example usage of the NN_PLOTTING_UTILITIES module.

This script demonstrates how to visualize neural networks with neurons
as circles and connections as lines.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.NN_DEFINITION_UTILITIES import (
    NeuralNetwork,
    FullyConnectedLayer,
    NetworkType
)
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig


def example_1_simple_network():
    """Example 1: Simple linear network."""
    print("=" * 70)
    print("Example 1: Simple Linear Network")
    print("=" * 70)
    
    # Create a simple network
    nn = NeuralNetwork(
        name="Simple Classifier",
        network_type=NetworkType.FEEDFORWARD,
        description="A basic 3-layer network"
    )
    
    # Add layers
    nn.add_layer(FullyConnectedLayer(num_neurons=4, name="Input"))
    nn.add_layer(FullyConnectedLayer(num_neurons=6, activation="relu", name="Hidden"))
    nn.add_layer(FullyConnectedLayer(num_neurons=3, activation="softmax", name="Output"))
    
    print(nn)
    print("\nPlotting network...")
    
    # Plot the network
    plot_network(
        nn,
        title="Simple 3-Layer Neural Network",
        save_path=os.path.join(os.path.dirname(__file__), "test_outputs", "output_simple_network.png"),
        show=True
    )
    
    print("✓ Plot saved as 'output_simple_network.png'")
    print()


def example_2_deeper_network():
    """Example 2: Deeper network with more layers."""
    print("=" * 70)
    print("Example 2: Deeper Network")
    print("=" * 70)
    
    nn = NeuralNetwork(
        name="Deep Classifier",
        network_type=NetworkType.FEEDFORWARD
    )
    
    # Create a deeper network
    nn.add_layer(FullyConnectedLayer(num_neurons=8, name="Input"))
    nn.add_layer(FullyConnectedLayer(num_neurons=10, activation="relu", name="Hidden 1"))
    nn.add_layer(FullyConnectedLayer(num_neurons=8, activation="relu", name="Hidden 2"))
    nn.add_layer(FullyConnectedLayer(num_neurons=6, activation="relu", name="Hidden 3"))
    nn.add_layer(FullyConnectedLayer(num_neurons=4, activation="softmax", name="Output"))
    
    print(nn)
    print("\nPlotting network...")
    
    # Plot with custom configuration
    config = PlotConfig(
        figsize=(14, 8),
        neuron_radius=0.25,
        connection_alpha=0.2,
        neuron_color='lightcoral'
    )
    
    plot_network(
        nn,
        title="Deep Neural Network (5 layers)",
        save_path=os.path.join(os.path.dirname(__file__), "test_outputs", "output_deep_network.png"),
        show=True,
        config=config
    )
    
    print("✓ Plot saved as 'output_deep_network.png'")
    print()


def example_3_branching_network():
    """Example 3: Branching network with multiple paths."""
    print("=" * 70)
    print("Example 3: Branching Network")
    print("=" * 70)
    
    nn = NeuralNetwork(
        name="Multi-Task Network",
        network_type=NetworkType.FEEDFORWARD,
        description="Network with branching structure"
    )
    
    # Shared backbone
    input_id = nn.add_layer(FullyConnectedLayer(num_neurons=6, name="Input"))
    shared_id = nn.add_layer(FullyConnectedLayer(num_neurons=8, activation="relu", name="Shared"))
    
    # Branch 1
    branch1_id = nn.add_layer(
        FullyConnectedLayer(num_neurons=4, activation="relu", name="Branch 1"),
        parent_ids=[shared_id]
    )
    output1_id = nn.add_layer(
        FullyConnectedLayer(num_neurons=2, activation="softmax", name="Output 1"),
        parent_ids=[branch1_id]
    )
    
    # Branch 2
    branch2_id = nn.add_layer(
        FullyConnectedLayer(num_neurons=4, activation="relu", name="Branch 2"),
        parent_ids=[shared_id]
    )
    output2_id = nn.add_layer(
        FullyConnectedLayer(num_neurons=3, activation="softmax", name="Output 2"),
        parent_ids=[branch2_id]
    )
    
    print(nn)
    print("\nPlotting branching network...")
    
    # Plot the branching network
    config = PlotConfig(
        figsize=(12, 10),
        neuron_color='lightgreen',
        connection_color='blue',
        connection_alpha=0.25
    )
    
    plot_network(
        nn,
        title="Multi-Task Branching Network",
        save_path=os.path.join(os.path.dirname(__file__), "test_outputs", "output_branching_network.png"),
        show=True,
        config=config
    )
    
    print("✓ Plot saved as 'output_branching_network.png'")
    print()


def example_4_mnist_network():
    """Example 4: MNIST-like network (with limited neuron display)."""
    print("=" * 70)
    print("Example 4: MNIST-like Network (Large Layers)")
    print("=" * 70)
    
    nn = NeuralNetwork(
        name="MNIST Classifier",
        network_type=NetworkType.FEEDFORWARD,
        description="Network for digit classification"
    )
    
    # Create a network with large layers
    nn.add_layer(FullyConnectedLayer(num_neurons=784, name="Input (28x28)"))
    nn.add_layer(FullyConnectedLayer(num_neurons=128, activation="relu", name="Hidden 1"))
    nn.add_layer(FullyConnectedLayer(num_neurons=64, activation="relu", name="Hidden 2"))
    nn.add_layer(FullyConnectedLayer(num_neurons=10, activation="softmax", name="Output"))
    
    print(nn)
    print("\nPlotting network (showing limited neurons for large layers)...")
    
    # Plot with max neurons per layer limit
    config = PlotConfig(
        figsize=(16, 8),
        max_neurons_per_layer=15,  # Limit display for large layers
        neuron_radius=0.2,
        connection_alpha=0.15,
        connection_linewidth=0.3
    )
    
    plot_network(
        nn,
        title="MNIST Digit Classifier",
        save_path=os.path.join(os.path.dirname(__file__), "test_outputs", "output_mnist_network.png"),
        show=True,
        config=config
    )
    
    print("✓ Plot saved as 'output_mnist_network.png'")
    print("  (Large layers show only 15 neurons as representatives)")
    print()


def example_5_multi_branch():
    """Example 5: Network with multiple branches from one layer."""
    print("=" * 70)
    print("Example 5: Multiple Children from Single Parent")
    print("=" * 70)
    
    nn = NeuralNetwork(
        name="Multi-Branch Network",
        network_type=NetworkType.FEEDFORWARD
    )
    
    # Input and shared layer
    input_id = nn.add_layer(FullyConnectedLayer(num_neurons=5, name="Input"))
    shared_id = nn.add_layer(FullyConnectedLayer(num_neurons=8, activation="relu", name="Shared"))
    
    # Create 3 branches
    for i in range(1, 4):
        branch_id = nn.add_layer(
            FullyConnectedLayer(num_neurons=4, activation="relu", name=f"Branch {i}"),
            parent_ids=[shared_id]
        )
        nn.add_layer(
            FullyConnectedLayer(num_neurons=2, activation="softmax", name=f"Output {i}"),
            parent_ids=[branch_id]
        )
    
    print(nn)
    print("\nPlotting multi-branch network...")
    
    config = PlotConfig(
        figsize=(12, 12),
        neuron_color='lightyellow',
        neuron_edge_color='darkorange',
        connection_color='purple',
        connection_alpha=0.3
    )
    
    plot_network(
        nn,
        title="Network with 3 Parallel Branches",
        save_path=os.path.join(os.path.dirname(__file__), "test_outputs", "output_multi_branch.png"),
        show=True,
        config=config
    )
    
    print("✓ Plot saved as 'output_multi_branch.png'")
    print()


def example_6_custom_styling():
    """Example 6: Custom styling options."""
    print("=" * 70)
    print("Example 6: Custom Styling")
    print("=" * 70)
    
    nn = NeuralNetwork(name="Styled Network")
    
    nn.add_layer(FullyConnectedLayer(num_neurons=5, name="Input"))
    nn.add_layer(FullyConnectedLayer(num_neurons=7, activation="relu", name="Hidden"))
    nn.add_layer(FullyConnectedLayer(num_neurons=3, activation="softmax", name="Output"))
    
    print(nn)
    print("\nPlotting with custom styling...")
    
    # Highly customized configuration
    config = PlotConfig(
        figsize=(10, 6),
        neuron_radius=0.4,
        layer_spacing=4.0,
        neuron_spacing=1.2,
        connection_alpha=0.5,
        connection_color='red',
        connection_linewidth=1.0,
        neuron_color='#FFD700',  # Gold
        neuron_edge_color='#8B0000',  # Dark red
        neuron_edge_width=2.0,
        show_neuron_labels=True,  # Show neuron indices
        title_fontsize=18,
        layer_name_fontsize=14
    )
    
    plot_network(
        nn,
        title="Custom Styled Neural Network",
        save_path=os.path.join(os.path.dirname(__file__), "test_outputs", "output_custom_style.png"),
        show=True,
        config=config
    )
    
    print("✓ Plot saved as 'output_custom_style.png'")
    print()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("NEURAL NETWORK PLOTTING EXAMPLES")
    print("="*70 + "\n")
    
    # Run all examples
    example_1_simple_network()
    example_2_deeper_network()
    example_3_branching_network()
    example_4_mnist_network()
    example_5_multi_branch()
    example_6_custom_styling()
    
    print("=" * 70)
    print("All examples completed!")
    print("Check the output PNG files in the current directory.")
    print("=" * 70)
