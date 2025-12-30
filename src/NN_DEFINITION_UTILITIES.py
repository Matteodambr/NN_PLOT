"""
NN_PLOTTING_UTILITIES - A module for plotting neural network architectures.

This module provides classes to represent and visualize neural network structures.
Currently supports feedforward neural networks with fully connected layers.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod


class NetworkType(Enum):
    """Enumeration of supported neural network types."""
    FEEDFORWARD = "feedforward"
    # Future types can be added here:
    # CONVOLUTIONAL = "convolutional"
    # RECURRENT = "recurrent"
    # TRANSFORMER = "transformer"


class LayerType(Enum):
    """Enumeration of supported layer types."""
    FULLY_CONNECTED = "fully_connected"
    INPUT_VECTOR = "input_vector"
    INPUT_IMAGE = "input_image"
    # Future types can be added here:
    # CONVOLUTIONAL = "convolutional"
    # POOLING = "pooling"
    # DROPOUT = "dropout"
    # BATCH_NORM = "batch_norm"


class Layer(ABC):
    """
    Abstract base class for all layer types.
    
    All layer implementations must inherit from this class and implement
    the required abstract methods.
    
    Attributes:
        layer_type (LayerType): The type of this layer.
        name (Optional[str]): Human-readable name for the layer.
        layer_id (str): Unique identifier for this layer (auto-generated).
    """
    
    def __init__(
        self,
        layer_type: LayerType,
        name: Optional[str] = None,
        layer_id: Optional[str] = None
    ):
        """
        Initialize a Layer object.
        
        Args:
            layer_type (LayerType): The type of this layer.
            name (Optional[str]): Human-readable name for the layer.
            layer_id (Optional[str]): Unique identifier. If None, auto-generates UUID.
        """
        self.layer_type = layer_type
        self.name = name
        
        # Generate a unique ID if not provided
        if layer_id is None:
            import uuid
            self.layer_id = str(uuid.uuid4())
        else:
            self.layer_id = layer_id
    
    @abstractmethod
    def get_output_size(self) -> int:
        """
        Get the output size (number of output units/neurons) of this layer.
        
        Returns:
            int: The output size of the layer.
        """
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """Return a human-readable string representation of the layer."""
        pass
    
    def __repr__(self) -> str:
        """Return a developer-friendly string representation of the layer."""
        return f"{self.__class__.__name__}(id='{self.layer_id[:8]}...', name='{self.name}')"


@dataclass
class FullyConnectedLayer(Layer):
    """
    Represents a fully connected (dense) layer in a neural network.
    
    A fully connected layer connects every neuron from the previous layer to every
    neuron in this layer. This is the most common type of layer in feedforward networks.
    
    Attributes:
        num_neurons (int): Number of neurons in this layer.
        activation (Optional[str]): Activation function name (e.g., 'relu', 'sigmoid', 'tanh', 'softmax').
        name (Optional[str]): Human-readable name for the layer.
        layer_id (str): Unique identifier for this layer (auto-generated).
        use_bias (bool): Whether this layer uses bias terms. Defaults to True.
        neuron_labels (Optional[List[str]]): Text labels for each neuron (supports LaTeX).
        label_position (str): Position of labels relative to neurons ('left' or 'right'). Default: 'left'.
    
    Example:
        >>> layer = FullyConnectedLayer(num_neurons=128, activation="relu", name="Hidden1")
        >>> print(layer.get_output_size())
        128
        >>> 
        >>> # Layer with LaTeX labels
        >>> input_layer = FullyConnectedLayer(
        ...     num_neurons=3,
        ...     name="Input",
        ...     neuron_labels=[r"$x_1$", r"$x_2$", r"$x_3$"],
        ...     label_position="left"
        ... )
    """
    num_neurons: int
    activation: Optional[str] = None
    name: Optional[str] = None
    layer_id: Optional[str] = None
    use_bias: bool = True
    neuron_labels: Optional[List[str]] = None
    label_position: str = "left"
    
    def __post_init__(self):
        """Validate the layer configuration after initialization."""
        if self.num_neurons <= 0:
            raise ValueError("Number of neurons must be positive")
        
        # Validate neuron_labels if provided
        if self.neuron_labels is not None:
            if len(self.neuron_labels) != self.num_neurons:
                raise ValueError(
                    f"Number of neuron_labels ({len(self.neuron_labels)}) must match "
                    f"num_neurons ({self.num_neurons})"
                )
        
        # Validate label_position
        if self.label_position not in ("left", "right"):
            raise ValueError("label_position must be 'left' or 'right'")
        
        # Initialize the base Layer class
        super().__init__(
            layer_type=LayerType.FULLY_CONNECTED,
            name=self.name,
            layer_id=self.layer_id
        )
    
    def get_output_size(self) -> int:
        """
        Get the output size of this fully connected layer.
        
        Returns:
            int: Number of neurons (output units) in this layer.
        """
        return self.num_neurons
    
    def __str__(self) -> str:
        """Return a human-readable string representation of the layer."""
        parts = [f"FullyConnected({self.num_neurons} neurons)"]
        if self.activation:
            parts.append(f"activation={self.activation}")
        if self.name:
            parts.append(f"name='{self.name}'")
        if not self.use_bias:
            parts.append("no_bias")
        return ", ".join(parts)
    
    def __repr__(self) -> str:
        """Return a developer-friendly string representation of the layer."""
        return (
            f"FullyConnectedLayer(neurons={self.num_neurons}, "
            f"activation={self.activation}, name='{self.name}', "
            f"id='{self.layer_id[:8]}...')"
        )


class InputLayer(Layer):
    """
    Abstract base class for all input layer types.
    
    Input layers are automatically treated as root layers (no parents) when added
    to a network. They represent the entry points for data into the network.
    
    Subclasses:
        - VectorInput: For 1D vector inputs (like tabular data)
        - ImageInput: For 2D/3D image inputs (future)
    """
    
    @property
    def is_input_layer(self) -> bool:
        """Mark this as an input layer for automatic root detection."""
        return True


@dataclass
class VectorInput(InputLayer):
    """
    Represents a vector (1D) input layer for the neural network.
    
    This is used for tabular/structured data where the input is a 1D vector of features.
    VectorInput layers are automatically treated as root layers (no parents).
    
    Attributes:
        num_features (int): Number of input features (vector dimensions).
        name (Optional[str]): Human-readable name for the layer.
        layer_id (str): Unique identifier for this layer (auto-generated).
        neuron_labels (Optional[List[str]]): Text labels for each neuron (supports LaTeX).
        label_position (str): Position of labels relative to neurons ('left' or 'right').
    
    Example:
        >>> input_layer = VectorInput(num_features=10, name="Input")
        >>> print(input_layer.get_output_size())
        10
        >>> 
        >>> # With LaTeX labels
        >>> input_layer = VectorInput(
        ...     num_features=3,
        ...     name="Features",
        ...     neuron_labels=[r"$x_1$", r"$x_2$", r"$x_3$"],
        ...     label_position="left"
        ... )
    """
    num_features: int
    name: Optional[str] = None
    layer_id: Optional[str] = None
    neuron_labels: Optional[List[str]] = None
    label_position: str = "left"
    
    def __post_init__(self):
        """Validate the layer configuration after initialization."""
        if self.num_features <= 0:
            raise ValueError("Number of features must be positive")
        
        # Validate neuron_labels if provided
        if self.neuron_labels is not None:
            if len(self.neuron_labels) != self.num_features:
                raise ValueError(
                    f"Number of neuron_labels ({len(self.neuron_labels)}) must match "
                    f"num_features ({self.num_features})"
                )
        
        # Validate label_position
        if self.label_position not in ("left", "right"):
            raise ValueError("label_position must be 'left' or 'right'")
        
        # Initialize the base Layer class
        super().__init__(
            layer_type=LayerType.INPUT_VECTOR,
            name=self.name,
            layer_id=self.layer_id
        )
    
    @property
    def num_neurons(self) -> int:
        """Alias for num_features to maintain compatibility with plotting."""
        return self.num_features
    
    def get_output_size(self) -> int:
        """
        Get the output size of this input layer.
        
        Returns:
            int: Number of features (output units) in this layer.
        """
        return self.num_features
    
    def __str__(self) -> str:
        """Return a human-readable string representation of the layer."""
        parts = [f"VectorInput({self.num_features} features)"]
        if self.name:
            parts.append(f"name='{self.name}'")
        return ", ".join(parts)
    
    def __repr__(self) -> str:
        """Return a developer-friendly string representation of the layer."""
        return (
            f"VectorInput(features={self.num_features}, name='{self.name}', "
            f"id='{self.layer_id[:8]}...')"
        )


class NeuralNetwork:
    """
    A class to represent and store neural network structure information.
    
    This class holds all relevant information about a neural network's architecture,
    including its layers, connections, name, and type. It serves as the foundation
    for visualizing neural network structures.
    
    The network supports both linear (sequential) and non-linear (branching) architectures
    through a parent-child relationship system between layers.
    
    Attributes:
        name (str): The name of the neural network.
        network_type (NetworkType): The type of neural network (e.g., FEEDFORWARD).
        layers (Dict[str, Layer]): Dictionary mapping layer IDs to Layer objects.
        connections (Dict[str, List[str]]): Maps parent layer IDs to lists of child layer IDs.
        description (Optional[str]): Optional description of the network.
        metadata (Dict[str, Any]): Additional metadata about the network.
    
    Example:
        >>> nn = NeuralNetwork(
        ...     name="My Classifier",
        ...     network_type=NetworkType.FEEDFORWARD
        ... )
        >>> input_id = nn.add_layer(FullyConnectedLayer(num_neurons=784, name="Input"))
        >>> hidden_id = nn.add_layer(FullyConnectedLayer(num_neurons=128, activation="relu", name="Hidden1"))
        >>> output_id = nn.add_layer(FullyConnectedLayer(num_neurons=10, activation="softmax", name="Output"))
    """
    
    def __init__(
        self,
        name: str,
        network_type: NetworkType = NetworkType.FEEDFORWARD,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a NeuralNetwork object.
        
        Args:
            name (str): The name of the neural network.
            network_type (NetworkType): The type of neural network. Defaults to FEEDFORWARD.
            description (Optional[str]): Optional description of the network.
            metadata (Optional[Dict[str, Any]]): Additional metadata about the network.
        """
        self.name = name
        self.network_type = network_type
        self.description = description
        self.metadata = metadata or {}
        self.layers: Dict[str, Layer] = {}  # layer_id -> Layer
        self.connections: Dict[str, List[str]] = {}  # parent_id -> [child_ids]
        self._layer_order: List[str] = []  # Maintains insertion order for linear networks
    
    def add_layer(
        self, 
        layer: Layer, 
        parent_ids: Optional[List[str]] = None,
        is_input: bool = False
    ) -> str:
        """
        Add a layer to the neural network.
        
        If parent_ids is None and the network already has layers, the new layer
        will be connected to the last added layer (linear/sequential structure).
        If parent_ids is provided, the layer will be connected to the specified parent(s).
        If the network is empty, the layer becomes the first layer with no parents.
        Use is_input=True to create an independent input layer (no parents) for multi-input networks.
        
        Args:
            layer (Layer): The layer object to add (e.g., FullyConnectedLayer).
            parent_ids (Optional[List[str]]): List of parent layer IDs. 
                If None and network is not empty, connects to the last layer.
                If None and network is empty, creates the first layer with no parents.
            is_input (bool): If True, creates an independent input layer with no parents.
                Use this for multi-input networks where you need multiple root layers.
                Equivalent to passing parent_ids=[].
        
        Returns:
            str: The unique ID of the added layer.
            
        Raises:
            ValueError: If a specified parent layer ID doesn't exist.
            TypeError: If layer is not an instance of Layer.
        """
        # Validate that layer is a Layer instance
        if not isinstance(layer, Layer):
            raise TypeError(f"Layer must be an instance of Layer, got {type(layer).__name__}")
        
        # Add the layer to the dictionary
        layer_id = layer.layer_id
        self.layers[layer_id] = layer
        self._layer_order.append(layer_id)
        
        # Handle connections
        # Automatically detect InputLayer types (VectorInput, ImageInput, etc.)
        is_input_layer = getattr(layer, 'is_input_layer', False)
        
        if is_input or is_input_layer:
            # Explicitly create an independent input layer (no parents)
            parent_ids = []
        elif parent_ids is None:
            # Linear structure: connect to the last layer if one exists
            if len(self._layer_order) > 1:
                parent_id = self._layer_order[-2]  # The previous layer
                parent_ids = [parent_id]
            else:
                # First layer, no parents
                parent_ids = []
        
        # Validate and create connections
        for parent_id in parent_ids:
            if parent_id not in self.layers:
                raise ValueError(f"Parent layer ID '{parent_id}' does not exist")
            
            # Add this layer as a child of the parent
            if parent_id not in self.connections:
                self.connections[parent_id] = []
            self.connections[parent_id].append(layer_id)
        
        return layer_id
    
    def get_parents(self, layer_id: str) -> List[str]:
        """
        Get the parent layer IDs for a given layer.
        
        Args:
            layer_id (str): The ID of the layer to query.
            
        Returns:
            List[str]: List of parent layer IDs.
            
        Raises:
            ValueError: If the layer ID doesn't exist.
        """
        if layer_id not in self.layers:
            raise ValueError(f"Layer ID '{layer_id}' does not exist")
        
        parents = []
        for parent_id, children in self.connections.items():
            if layer_id in children:
                parents.append(parent_id)
        return parents
    
    def get_children(self, layer_id: str) -> List[str]:
        """
        Get the child layer IDs for a given layer.
        
        Args:
            layer_id (str): The ID of the layer to query.
            
        Returns:
            List[str]: List of child layer IDs.
            
        Raises:
            ValueError: If the layer ID doesn't exist.
        """
        if layer_id not in self.layers:
            raise ValueError(f"Layer ID '{layer_id}' does not exist")
        
        return self.connections.get(layer_id, [])
    
    def get_root_layers(self) -> List[str]:
        """
        Get all root layers (layers with no parents).
        
        Returns:
            List[str]: List of root layer IDs.
        """
        all_children = set()
        for children in self.connections.values():
            all_children.update(children)
        
        root_layers = [layer_id for layer_id in self.layers.keys() 
                      if layer_id not in all_children]
        return root_layers
    
    def has_input_layer(self) -> bool:
        """
        Check if the network has at least one input layer.
        
        An input layer is defined as either:
        - A layer that inherits from InputLayer (e.g., VectorInput)
        - A root layer (layer with no parents)
        
        Returns:
            bool: True if the network has at least one input layer.
        """
        root_layers = self.get_root_layers()
        return len(root_layers) > 0
    
    def get_input_layers(self) -> List[str]:
        """
        Get all input layers in the network.
        
        Input layers are root layers (layers with no parents), which includes
        both VectorInput layers and any FullyConnectedLayer added as the first
        layer or with is_input=True.
        
        Returns:
            List[str]: List of input layer IDs.
        """
        return self.get_root_layers()
    
    def get_leaf_layers(self) -> List[str]:
        """
        Get all leaf layers (layers with no children).
        
        Returns:
            List[str]: List of leaf layer IDs.
        """
        return [layer_id for layer_id in self.layers.keys() 
                if layer_id not in self.connections or len(self.connections[layer_id]) == 0]
    
    def remove_layer(self, layer_id: str) -> None:
        """
        Remove a layer from the neural network by its ID.
        
        This will also remove all connections involving this layer.
        
        Args:
            layer_id (str): The ID of the layer to remove.
            
        Raises:
            ValueError: If the layer ID doesn't exist.
        """
        if layer_id not in self.layers:
            raise ValueError(f"Layer ID '{layer_id}' does not exist")
        
        # Remove the layer
        del self.layers[layer_id]
        self._layer_order.remove(layer_id)
        
        # Remove from connections (as parent)
        if layer_id in self.connections:
            del self.connections[layer_id]
        
        # Remove from connections (as child)
        for parent_id in list(self.connections.keys()):
            if layer_id in self.connections[parent_id]:
                self.connections[parent_id].remove(layer_id)
                # Clean up empty connection lists
                if not self.connections[parent_id]:
                    del self.connections[parent_id]
    
    def get_layer(self, layer_id: str) -> Layer:
        """
        Get layer object by ID.
        
        Args:
            layer_id (str): The ID of the layer to retrieve.
            
        Returns:
            Layer: The layer object.
            
        Raises:
            ValueError: If the layer ID doesn't exist.
        """
        if layer_id not in self.layers:
            raise ValueError(f"Layer ID '{layer_id}' does not exist")
        return self.layers[layer_id]
    
    def get_layer_by_name(self, name: str) -> Optional[Layer]:
        """
        Get layer object by name.
        
        Args:
            name (str): The name of the layer to retrieve.
            
        Returns:
            Optional[Layer]: The layer object, or None if not found.
        """
        for layer in self.layers.values():
            if layer.name == name:
                return layer
        return None
    
    def get_layer_id_by_name(self, name: str) -> Optional[str]:
        """
        Get layer ID by name.
        
        Args:
            name (str): The name of the layer.
            
        Returns:
            Optional[str]: The layer ID, or None if not found.
        """
        for layer_id, layer in self.layers.items():
            if layer.name == name:
                return layer_id
        return None
    
    def num_layers(self) -> int:
        """
        Get the total number of layers in the network.
        
        Returns:
            int: The number of layers.
        """
        return len(self.layers)
    
    def get_total_neurons(self) -> int:
        """
        Calculate the total number of neurons across all layers.
        
        For fully connected layers, this counts the number of neurons.
        For other layer types, it counts the output size.
        
        Returns:
            int: Total number of output units across all layers.
        """
        return sum(layer.get_output_size() for layer in self.layers.values())
    
    def is_linear(self) -> bool:
        """
        Check if the network has a linear (sequential) structure.
        
        A network is considered linear if each layer (except the first) has exactly
        one parent, and each layer (except the last) has exactly one child.
        
        Returns:
            bool: True if the network is linear, False otherwise.
        """
        if len(self.layers) == 0:
            return True
        
        # Check that there's exactly one root
        roots = self.get_root_layers()
        if len(roots) != 1:
            return False
        
        # Check that each layer has at most one child and one parent
        for layer_id in self.layers.keys():
            children = self.get_children(layer_id)
            parents = self.get_parents(layer_id)
            
            if len(children) > 1 or len(parents) > 1:
                return False
        
        return True
    
    def __repr__(self) -> str:
        """Return a string representation of the neural network."""
        return (
            f"NeuralNetwork(name='{self.name}', "
            f"type={self.network_type.value}, "
            f"layers={self.num_layers()}, "
            f"total_neurons={self.get_total_neurons()}, "
            f"linear={self.is_linear()})"
        )
    
    def __str__(self) -> str:
        """Return a human-readable string representation of the neural network."""
        lines = [
            f"Neural Network: {self.name}",
            f"Type: {self.network_type.value}",
            f"Structure: {'Linear' if self.is_linear() else 'Non-linear (branching)'}",
            f"Total Layers: {self.num_layers()}",
            f"Total Output Units: {self.get_total_neurons()}"
        ]
        
        if self.description:
            lines.append(f"Description: {self.description}")
        
        if self.layers:
            lines.append("\nLayers:")
            for i, layer_id in enumerate(self._layer_order):
                layer = self.layers[layer_id]
                
                # Build layer description
                layer_str = f"  Layer {i}: "
                
                # Use the layer's __str__ method for type-specific info
                if isinstance(layer, FullyConnectedLayer):
                    layer_str += f"{layer.get_output_size()} neurons"
                    if layer.name:
                        layer_str += f" ('{layer.name}')"
                    if layer.activation:
                        layer_str += f" - activation: {layer.activation}"
                    layer_str += f" - type: {layer.layer_type.value}"
                else:
                    # For future layer types
                    layer_str += str(layer)
                
                # Show parent information
                parents = self.get_parents(layer_id)
                if parents:
                    parent_names = []
                    for parent_id in parents:
                        parent_layer = self.layers[parent_id]
                        parent_names.append(parent_layer.name if parent_layer.name else f"Layer {self._layer_order.index(parent_id)}")
                    layer_str += f" <- parents: [{', '.join(parent_names)}]"
                
                # Show children information
                children = self.get_children(layer_id)
                if children:
                    child_names = []
                    for child_id in children:
                        child_layer = self.layers[child_id]
                        child_names.append(child_layer.name if child_layer.name else f"Layer {self._layer_order.index(child_id)}")
                    layer_str += f" -> children: [{', '.join(child_names)}]"
                
                lines.append(layer_str)
        
        return "\n".join(lines)
    
    def summary(self) -> str:
        """
        Generate a summary of the neural network architecture.
        
        Returns:
            str: A formatted summary of the network.
        """
        return str(self)
