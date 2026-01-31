# NeuralNet - Visualizer

## A Python Module for Visualizing Neural Network Architectures

NeuralNet-Visualizer is a powerful Python module for representing and visualizing custom neural network architectures. It provides a flexible class structure to model sequential networks as well as multi-head, multi-input, and skip connection topologies with highly customizable visualizations.

![NeuralNet-Visualizer Showcase](src/readme_image_static/showcase.png)

### Usage & Attribution

This project is open source and free to use. If you use NeuralNet-Visualizer in your work, please provide appropriate credit by citing this repository.

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Matteodambr/NeuralNet-Visualizer.git
cd NeuralNet-Visualizer
pip install matplotlib
```

---

## Summary of Features Demonstrated

| Demo | Feature | Description |
|------|---------|-------------|
| **Demo 1** | Basic Setup | Importing modules and configuring the environment |
| **Demo 2** | Basic Network Creation | Creating `NeuralNetwork` and `FullyConnectedLayer` objects, background color |
| **Demo 3** | Multi-Input/Output Networks | Using `parent_ids` for multi-head, multi-input, and skip connections |
| **Demo 4** | Layer-Specific Styling | Custom colors, boxes, and connections via `LayerStyle` |
| **Demo 5** | Spacing and Layout | Controlling layer spacing, neuron spacing, and branch spacing |
| **Demo 6** | Complete Labeling | Neuron labels (LaTeX), layer labels, group brackets, variable names |
| **Demo 7** | MLP Customization | Neuron collapsing for large layers with ellipsis notation |
| **Demo 8** | CNN Customization | ImageInput layer with text, image, and RGB channel modes |
| **Demo 9** | RNN Customization | 🚧 Work in progress |

### Key Components

**Network Container**
- **`NeuralNetwork`**: Container for network architecture

**Input Layers**
- **`VectorInput`**: For tabular/vector input data
- **`ImageInput`**: For image data with multiple display modes (text, single image, RGB channels)
- 🚧 `SequenceInput`: For sequential/time-series data (work in progress)

**Intermediate Layers**
- **`FullyConnectedLayer`**: Dense layer with neurons, activation, and optional labels
- 🚧 `ConvolutionalLayer`: CNN-based layers (work in progress)
- 🚧 `RecurrentLayer`: RNN/LSTM/GRU layers (work in progress)

**Customization**
- **`PlotConfig`**: Configuration for all visualization options
- **`LayerStyle`**: Per-layer styling (colors, boxes, collapsing)
- **`LayerGroup`**: Bracket grouping for related layers

For more details, explore the docstrings in `NN_PLOTTING_UTILITIES.py`.

---

## Demo 1: Basic Setup

First, let's import the necessary modules. The library consists of two main components:
- `NN_DEFINITION_UTILITIES`: Classes for defining neural network structures
- `NN_PLOTTING_UTILITIES`: Functions and classes for visualization

```python
import sys
import os
import importlib

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

# Import and reload the modules to pick up any changes
import NN_DEFINITION_UTILITIES
import NN_PLOTTING_UTILITIES
importlib.reload(NN_DEFINITION_UTILITIES)
importlib.reload(NN_PLOTTING_UTILITIES)

# Import the main components
from NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer, VectorInput
from NN_PLOTTING_UTILITIES import plot_network, PlotConfig, LayerGroup, LayerStyle

# Configure matplotlib for inline display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
%matplotlib inline

print("✓ All modules imported successfully!")
```

---

## Demo 2: Creating a Simple Network

Let's start by creating a basic feedforward neural network with an input layer, hidden layer, and output layer. We'll also demonstrate the `background_color` and font options.

**Layer Types:**
- **`VectorInput`**: For input layers - automatically treated as root layers (no parents needed)
- **`FullyConnectedLayer`**: For hidden and output layers with optional activation functions

**Appearance Options:**
- `background_color`: Set to `"None"` for transparent, or any color name/RGB value
- `font_family`: Font for all text (default: `"Times New Roman"`)

**Font Size Parameters:**
| Label Type | Parameter | Default |
|------------|-----------|---------|
| Title | `title_fontsize` | 16 |
| Neuron labels | `neuron_text_label_fontsize` | 10 |
| Layer names | `layer_name_fontsize` | 12 |
| Variable names | `layer_variable_names_fontsize` | 11 |
| Group brackets | `label_fontsize` (in `LayerGroup`) | 12 |

```python
# Create a simple feedforward network
simple_nn = NeuralNetwork("Simple Feedforward Network")

# Add layers sequentially
# VectorInput is for input layers - automatically treated as a root layer
simple_nn.add_layer(VectorInput(num_features=4, name="Input"))
simple_nn.add_layer(FullyConnectedLayer(num_neurons=6, activation="relu", name="Hidden"))
simple_nn.add_layer(FullyConnectedLayer(num_neurons=2, activation="softmax", name="Output"))

# Plot the network with custom appearance settings
fig = plot_network(
    simple_nn, 
    title="Simple Feedforward Network",
    show=True,
    config=PlotConfig(
        figsize=(6, 4),
        
        # Background color options:
        background_color="lightgray",     # Current setting
        # background_color="white",       # White background
        # background_color="None",        # Transparent background
        # background_color="#F5F5F5",     # Custom hex color
        
        # Font family options:
        font_family="Times New Roman",    # Current setting (serif)
        # font_family="DejaVu Sans",      # Sans-serif option
        # font_family="Arial",            # Another sans-serif
        # font_family="Courier New",      # Monospace option
        
        title_fontsize=14
    )
)
```

    
![png](src/readme_image_static/README_7_0.png)
    

---

## Demo 3: Multi-Input and Multi-Output Networks

The library supports non-sequential architectures through explicit parent-child relationships using `parent_ids`. This enables:
- **Multi-output**: One layer feeding into multiple downstream layers (multi-head networks)
- **Multi-input**: Multiple layers merging into a single downstream layer (fusion networks)
- **Skip connections**: Layers connecting to non-adjacent layers

```python
# Create a network with both multi-input and multi-output
multi_in_multi_out_nn = NeuralNetwork("Multi-Input Multi-Output Network")

# Two separate input streams using VectorInput (automatically treated as root layers)
input1_id = multi_in_multi_out_nn.add_layer(VectorInput(num_features=3, name="Input A"))
input2_id = multi_in_multi_out_nn.add_layer(VectorInput(num_features=3, name="Input B"))

# Fusion layer receiving from both inputs
fusion_id = multi_in_multi_out_nn.add_layer(
    FullyConnectedLayer(num_neurons=6, activation="relu", name="Fusion"), 
    parent_ids=[input1_id, input2_id]  # Multiple parents!
)

# Two output heads branching from the fusion layer (multi-output)
output1_id = multi_in_multi_out_nn.add_layer(
    FullyConnectedLayer(num_neurons=2, activation="softmax", name="Classification"), 
    parent_ids=[fusion_id]
)
output2_id = multi_in_multi_out_nn.add_layer(
    FullyConnectedLayer(num_neurons=1, activation="linear", name="Regression"), 
    parent_ids=[fusion_id]
)

fig = plot_network(
    multi_in_multi_out_nn, 
    title="Multi-Input Multi-Output Network",
    config=PlotConfig(figsize=(12, 6), show_layer_names=True, layer_names_show_activation=True),
    show=True
)
```

    
![png](src/readme_image_static/README_9_0.png)
    

---

## Demo 4: Layer-Specific Styling

Each layer can have its own visual style using `LayerStyle`. This includes custom colors for neurons and connections, boxes around layers, and more. Here we style the multi-output network from Demo 3.

```python
# Define custom styles for each layer in the multi-input multi-output network
mimo_styles = {
    "Input A": LayerStyle(
        neuron_fill_color="#E3F2FD",  # Light blue
        neuron_edge_color="#1976D2",   # Blue
        neuron_edge_width=2.5,
        connection_color="#1976D2",
        connection_alpha=0.5,
        box_around_layer=True,        # Draw a box around this layer
        box_fill_color="#BBDEFB",
        box_edge_color="#1976D2",
        box_edge_width=2.0,
        box_padding=0.5,
        box_corner_radius=0.3
    ),
    "Input B": LayerStyle(
        neuron_fill_color="#E3F2FD",  # Light blue
        neuron_edge_color="#1976D2",   # Blue
        neuron_edge_width=2.5,
        connection_color="#1976D2",
        connection_alpha=0.5
    ),
    "Fusion": LayerStyle(
        neuron_fill_color="#E8F5E9",  # Light green
        neuron_edge_color="#388E3C",   # Green
        neuron_edge_width=2.5,
        connection_color="#388E3C",
        connection_alpha=0.5
    ),
    "Classification": LayerStyle(
        neuron_fill_color="#E8F5E9",
        neuron_edge_color="#2E7D32",
        neuron_edge_width=2.5,
        box_around_layer=True,        # Draw a box around this layer
        box_fill_color="#C8E6C9",
        box_edge_color="#2E7D32",
        box_edge_width=2.0,
        box_padding=0.5,
        box_corner_radius=0.3
    ),
    "Regression": LayerStyle(
        neuron_fill_color="#FFF3E0",
        neuron_edge_color="#E65100",
        neuron_edge_width=2.5
    )
}

styled_config = PlotConfig(
    figsize=(12, 6),
    layer_styles=mimo_styles,
    show_layer_names=True,    # Hide layer name labels
    branch_spacing=4.5         # Increase spacing between Classification/Regression layers
)

fig = plot_network(multi_in_multi_out_nn, title="Styled Multi-Input Multi-Output Network", config=styled_config, show=True)
```

    
![png](src/readme_image_static/README_11_0.png)
    

---

## Demo 5: Spacing and Layout Control

Control the overall layout and spacing of your network visualization using `PlotConfig` parameters:

| Parameter | Description |
|-----------|-------------|
| `layer_spacing` | Horizontal distance between adjacent layers |
| `neuron_spacing` | Vertical distance between neurons within a layer |
| `branch_spacing` | Vertical distance between layers that branch from the same parent |
| `neuron_radius` | Size of each neuron circle |
| `figsize` | Overall figure dimensions (width, height) |

```python
# Demonstrate spacing controls with side-by-side comparisons
import matplotlib.pyplot as plt

# --- Layer Spacing Comparison ---
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# When ax is provided, show is automatically forced to False
plot_network(
    simple_nn,
    title="Default Layer Spacing",
    config=PlotConfig(figsize=(6, 4)),
    ax=axes[0]
)

plot_network(
    simple_nn,
    title="Increased (layer_spacing=4.0)",
    config=PlotConfig(figsize=(6, 4), layer_spacing=4.0),
    ax=axes[1]
)

plt.tight_layout()
plt.show()

# --- Neuron Spacing Comparison ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_network(
    simple_nn,
    title="Default Neuron Spacing",
    config=PlotConfig(figsize=(6, 4)),
    ax=axes[0]
)

plot_network(
    simple_nn,
    title="Increased (neuron_spacing=1.5)",
    config=PlotConfig(figsize=(6, 5), neuron_spacing=1.5),
    ax=axes[1]
)

plt.tight_layout()
plt.show()

# --- Branch Spacing Comparison ---
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

plot_network(
    multi_in_multi_out_nn,
    title="Default Branch Spacing",
    config=PlotConfig(figsize=(8, 5)),
    ax=axes[0]
)

plot_network(
    multi_in_multi_out_nn,
    title="Increased (branch_spacing=5.0)",
    config=PlotConfig(figsize=(8, 5), branch_spacing=5.0),
    ax=axes[1]
)

plt.tight_layout()
plt.show()
```

    
![png](src/readme_image_static/README_13_0.png)
    

    
![png](src/readme_image_static/README_13_1.png)
    

    
![png](src/readme_image_static/README_13_2.png)
    

---

## Demo 6: Complete Labeling Configuration

This section demonstrates all labeling features on an autoencoder architecture:

- **Neuron labels**: LaTeX labels on individual neurons (input, latent, output)
- **Layer name labels**: Configurable labels below each layer with toggleable components:
  - `layer_names_show_dim=True/False` — show/hide neuron count
  - `layer_names_show_activation=True/False` — show/hide activation function
  - `layer_names_show_type=True/False` — show/hide layer type (e.g., "FullyConnected")
- **Layer group brackets**: Curly braces to group related layers (Encoder, Decoder)
- **Variable name labels**: High-level descriptions for key layers
- **Layer collapsing**: Automatic ellipsis notation for large layers

```python
# Create an autoencoder network for dimensionality reduction
autoencoder = NeuralNetwork("Autoencoder for Dimensionality Reduction")

# Input layer with LaTeX labels using VectorInput
# Labels go from x_256 (top) to x_1 (bottom)
input_id = autoencoder.add_layer(VectorInput(
    num_features=256,
    name="Input Layer",
    neuron_labels=[r"$x_{" + str(i) + "}$" for i in range(256, 0, -1)],
    label_position="left"
))

# Encoder layers
enc1_id = autoencoder.add_layer(FullyConnectedLayer(
    num_neurons=128, activation="relu", name="Encoder 1"
), parent_ids=[input_id])

enc2_id = autoencoder.add_layer(FullyConnectedLayer(
    num_neurons=4, activation="relu", name="Encoder 2"
), parent_ids=[enc1_id])

# Latent space (bottleneck) with labeled neurons
latent_id = autoencoder.add_layer(FullyConnectedLayer(
    num_neurons=2,
    activation="linear",
    name="Latent Space",
    neuron_labels=[r"$z_2$", r"$z_1$"],
    label_position="right"
), parent_ids=[enc2_id])

# Decoder layers
dec1_id = autoencoder.add_layer(FullyConnectedLayer(
    num_neurons=4, activation="relu", name="Decoder 1"
), parent_ids=[latent_id])

dec2_id = autoencoder.add_layer(FullyConnectedLayer(
    num_neurons=128, activation="relu", name="Decoder 2"
), parent_ids=[dec1_id])

# Output layer with LaTeX labels - matches input but with hat notation
output_id = autoencoder.add_layer(FullyConnectedLayer(
    num_neurons=256,
    activation="sigmoid",
    name="Output Layer",
    neuron_labels=[r"$\hat{x}_{" + str(i) + "}$" for i in range(256, 0, -1)],
    label_position="right"
), parent_ids=[dec2_id])

print("✓ Autoencoder network created")
```

```python
# Define comprehensive layer styles
autoencoder_styles = {
    'Input Layer': LayerStyle(
        neuron_fill_color='#E8F4F8',
        neuron_edge_color='#1976D2',
        neuron_edge_width=2.5,
        box_around_layer=True,
        box_fill_color='#E3F2FD',
        box_edge_color='#1976D2',
        box_edge_width=2.5,
        box_padding=0.6,
        box_corner_radius=0.4,
        box_include_neuron_labels=True,
        max_neurons_to_plot=10,
        collapse_neurons_start=4,
        collapse_neurons_end=4,
        layer_name_bold=True,
        variable_name_color='#E3F2FD'
    ),
    'Encoder 1': LayerStyle(
        neuron_fill_color='#E8F4F8',
        neuron_edge_color='#1976D2',
        max_neurons_to_plot=10,
        collapse_neurons_start=3,
        collapse_neurons_end=3,
        layer_name_bold=True
    ),
    'Encoder 2': LayerStyle(
        neuron_fill_color='#E8F4F8',
        neuron_edge_color='#1976D2',
        max_neurons_to_plot=10,
        collapse_neurons_start=2,
        collapse_neurons_end=2,
        layer_name_bold=True
    ),
    'Latent Space': LayerStyle(
        neuron_fill_color='#FFF3E0',
        neuron_edge_color='#F57C00',
        neuron_edge_width=3.0,
        layer_name_bold=True,
        variable_name_color='#FFF3E0'
    ),
    'Decoder 1': LayerStyle(
        neuron_fill_color='#FFEBEE',
        neuron_edge_color='#D32F2F',
        max_neurons_to_plot=10,
        collapse_neurons_start=2,
        collapse_neurons_end=2,
        layer_name_bold=True
    ),
    'Decoder 2': LayerStyle(
        neuron_fill_color='#FFEBEE',
        neuron_edge_color='#D32F2F',
        max_neurons_to_plot=10,
        collapse_neurons_start=3,
        collapse_neurons_end=3,
        layer_name_bold=True
    ),
    'Output Layer': LayerStyle(
        neuron_fill_color='#FCE4EC',
        neuron_edge_color='#C2185B',
        neuron_edge_width=2.5,
        box_around_layer=True,
        box_fill_color='#F8BBD0',
        box_edge_color='#C2185B',
        box_edge_width=2.5,
        box_padding=0.6,
        box_corner_radius=0.4,
        box_include_neuron_labels=True,
        max_neurons_to_plot=10,
        collapse_neurons_start=4,
        collapse_neurons_end=4,
        layer_name_bold=True,
        variable_name_color='#F8BBD0'
    )
}

# Define layer groups with curly brackets
autoencoder_groups = [
    LayerGroup(
        layer_ids=['Input Layer', 'Encoder 1', 'Encoder 2'],
        label='Encoder Network',
        bracket_style='curly',
        bracket_color='#1976D2',
        bracket_linewidth=2.5,
        bracket_height=0.4,
        label_fontsize=16,
        label_color='#1976D2',
        y_offset=-2.0
    ),
    LayerGroup(
        layer_ids=['Decoder 1', 'Decoder 2', 'Output Layer'],
        label='Decoder Network',
        bracket_style='curly',
        bracket_color='#D32F2F',
        bracket_linewidth=2.5,
        bracket_height=0.4,
        label_fontsize=16,
        label_color='#D32F2F',
        y_offset=-2.0
    )
]

print("✓ Layer styles and groups configured")
```

```python
# Create the comprehensive labeling configuration
label_config = PlotConfig(
    # Figure settings
    figsize=(16, 10),
    background_color='white',
    font_family='Times New Roman',
    
    # Neuron appearance
    neuron_radius=0.35,
    neuron_edge_width=2.0,
    
    # Neuron text labels (LaTeX labels on neurons)
    show_neuron_text_labels=True,
    neuron_text_label_fontsize=18,
    neuron_text_label_offset=0.85,
    
    # Connections
    connection_alpha=0.65,
    connection_color='gray',
    connection_linewidth=2.0,
    
    # Neuron collapse settings
    max_neurons_per_layer=10,
    collapse_neurons_start=4,
    collapse_neurons_end=4,
    
    # Layer names (below each layer)
    show_layer_names=True,
    layer_name_fontsize=12,
    layer_names_show_type=False,        # Hide layer type
    layer_names_show_dim=True,          # Show neuron count
    layer_names_show_activation=True,   # Show activation
    layer_names_align_bottom=True,
    layer_names_bottom_offset=1.8,
    
    # Layer variable names (high-level descriptions)
    show_layer_variable_names=True,
    layer_variable_names={
        'Input Layer': 'Input\nFeatures',
        'Latent Space': 'Latent\nRepresentation',
        'Output Layer': 'Reconstructed\nOutput'
    },
    layer_variable_names_fontsize=13,
    layer_variable_names_position='side',
    layer_variable_names_multialignment='center',
    layer_variable_names_offset=1.2,
    
    # Title
    title_fontsize=18,
    
    # Layer-specific styling
    layer_styles=autoencoder_styles,
    
    # Layer group brackets
    layer_groups=autoencoder_groups
)

# Generate the plot
fig = plot_network(
    autoencoder,
    config=label_config,
    title="Autoencoder Architecture: Complete Feature Showcase",
    show=True
)
```

    
![png](src/readme_image_static/README_17_0.png)
    

---

## Demo 7: MLP-Specific Customization

For Multi-Layer Perceptrons (MLPs) with many neurons, you can customize how large layers are collapsed with ellipsis notation. This keeps visualizations readable while preserving the network structure.

```python
# Create a large MLP network
large_mlp = NeuralNetwork("Large MLP")

large_mlp.add_layer(VectorInput(
    num_features=256, 
    name="Input",
    neuron_labels=[f"$x_{{{i}}}$" for i in range(256, 0, -1)],
    label_position="left"
))
large_mlp.add_layer(FullyConnectedLayer(num_neurons=128, activation="relu", name="Hidden 1"))
large_mlp.add_layer(FullyConnectedLayer(num_neurons=64, activation="relu", name="Hidden 2"))
large_mlp.add_layer(FullyConnectedLayer(num_neurons=32, activation="relu", name="Hidden 3"))
large_mlp.add_layer(FullyConnectedLayer(
    num_neurons=10, 
    activation="softmax", 
    name="Output",
    neuron_labels=[f"$y_{{{i}}}$" for i in range(10, 0, -1)],
    label_position="right"
))

# Configure collapsing behavior
mlp_config = PlotConfig(
    figsize=(16, 10),
    
    # Global collapsing settings
    max_neurons_per_layer=10,      # Maximum neurons before collapsing
    collapse_neurons_start=4,       # Show first N neurons
    collapse_neurons_end=4,         # Show last N neurons
    
    # Per-layer collapsing overrides via LayerStyle
    layer_styles={
        "Hidden 2": LayerStyle(
            max_neurons_to_plot=8,
            collapse_neurons_start=3,
            collapse_neurons_end=3,
            neuron_fill_color="#E8F5E9",
            neuron_edge_color="#388E3C"
        )
    },
    
    # Show neuron labels on input/output
    show_neuron_text_labels=True,
    neuron_text_label_fontsize=14,
    
    # Neuron numbering (shows index inside each neuron)
    show_neuron_labels=True,                 # Display neuron indices
    neuron_numbering_reversed=True,          # N-1 at top, 0 at bottom (matches array indexing)
    neuron_label_fontsize=12,                # Font size for neuron numbers
    
    # Layer names
    show_layer_names=True,
    layer_names_show_dim=True,
    layer_names_show_activation=True
)

fig = plot_network(large_mlp, title="Large MLP with Neuron Collapsing and Numbering", config=mlp_config, show=True)
```

    
![png](src/readme_image_static/README_19_0.png)
    

---

## Demo 8: CNN-Specific Customization

CNN networks now support the **`ImageInput`** layer, which provides flexible visualization options for image inputs.

### ImageInput Features

The `ImageInput` layer supports three display modes:

1. **Text Mode**: Display a rounded rectangle with dimension text
2. **Single Image Mode**: Display an actual image from a file or URL
3. **RGB Channels Mode**: Separate RGB channels into 3 overlapped rectangles

```python
from NN_DEFINITION_UTILITIES import ImageInput, NeuralNetwork, FullyConnectedLayer
from NN_PLOTTING_UTILITIES import NetworkPlotter, PlotConfig

# Example 1: Text mode (default - no image file needed)
network = NeuralNetwork(name="CNN with Text Input")

img_input = ImageInput(
    height=224,
    width=224,
    channels=3,
    name="Input Image",
    display_mode="text",
    custom_text="224 x 224 x 3",  # Optional custom text
    custom_text_size=14
)
network.add_layer(img_input, is_input=True)

# Add CNN layers
conv1 = FullyConnectedLayer(num_neurons=64, activation="ReLU", name="Conv1")
network.add_layer(conv1, parent_ids=[img_input.layer_id])

output = FullyConnectedLayer(num_neurons=10, activation="Softmax", name="Output")
network.add_layer(output, parent_ids=[conv1.layer_id])

# Plot
config = PlotConfig(figsize=(10, 8))
plotter = NetworkPlotter(config)
plotter.plot_network(network, title="CNN with ImageInput", show=True)
```

### Advanced ImageInput Features

```python
# Example 2: Display actual image with magnification
img_input = ImageInput(
    height=224,
    width=224,
    channels=3,
    display_mode="single_image",
    image_path="path/to/image.jpg",  # Local file or URL
    magnification=1.5,                # Zoom in 1.5x
    translation_x=0.2,                # Shift right
    translation_y=-0.1,               # Shift up
    rounded_corners=True              # Rounded rectangle (default)
)

# Example 3: Black & white conversion
img_input = ImageInput(
    height=224,
    width=224,
    channels=1,
    display_mode="single_image",
    image_path="color_image.jpg",
    color_mode="bw"  # Convert to grayscale
)

# Example 4: RGB channel separation
img_input = ImageInput(
    height=224,
    width=224,
    channels=3,
    display_mode="rgb_channels",
    image_path="image.jpg"  # Shows 3 overlapped rectangles
)
```

### ImageInput Parameters

- **Basic dimensions**: `height`, `width`, `channels` (1 for BW, 3 for RGB)
- **Display mode**: `"text"`, `"single_image"`, or `"rgb_channels"`
- **Text options**: `custom_text`, `custom_text_size`
- **Image options**: `image_path` (local file or URL)
- **Transforms**: `magnification` (zoom factor), `translation_x`, `translation_y` (offset from center)
- **Styling**: `rounded_corners` (bool), `corner_radius`, `color_mode` ("rgb" or "bw")

For complete examples, see `src/tests/demo_image_input.py`.

---

## Demo 9: Recurrent Network Customization

> ⚠️ **Work in Progress**: Recurrent network visualization is under development. Future versions will include:
> - LSTM/GRU cell representations
> - Recurrent connection arrows
> - Sequence unrolling visualization
> - Hidden state indicators

```python
# Placeholder for RNN visualization
print("🚧 Recurrent network visualization coming soon!")
print("   Planned features:")
print("   - LSTM cells with gate indicators")
print("   - GRU cells")
print("   - Bidirectional RNN arrows")
print("   - Sequence length annotations")
```

    🚧 Recurrent network visualization coming soon!
       Planned features:
       - LSTM cells with gate indicators
       - GRU cells
       - Bidirectional RNN arrows
       - Sequence length annotations

---

## Saving Your Visualization

You can save your plots in multiple formats (PNG, SVG, PDF) with custom DPI settings.

```python
# Create output directory
os.makedirs("showcase_outputs", exist_ok=True)

# Save as PNG with high DPI
plot_network(
    autoencoder,
    config=label_config,
    title="Autoencoder: Complete Labeling Showcase",
    save_path="showcase_outputs/autoencoder_labeling.png",
    show=False,
    dpi=300
)
print("✓ Saved: showcase_outputs/autoencoder_labeling.png")
```

    
![png](src/readme_image_static/README_25_1.png)
    

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
