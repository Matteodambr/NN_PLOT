# NeuralNet - Visualizer

## Overview

NeuralNet - Visualizer is a Python module for representing and visualizing custom neural network architectures. It provides a flexible class structure to model both linear (sequential) and non-linear (branching) neural network topologies with highly customizable visualizations.

<img width="4612" height="2970" alt="encoder_decoder_comprehensive" src="https://github.com/user-attachments/assets/46a841dd-0f35-4187-a522-86cbfefcd88b" />

## Features

- **Flexible Network Structure**: Support for both linear and branching architectures
- **Parent-Child Relationships**: Explicit control over layer connections
- **Beautiful Visualizations**: Plot networks with neurons as circles and connections as lines
- **Layer-Specific Styling**: Customize colors, edge styles, and line widths for each layer
- **Rounded Layer Boxes**: Draw boxes around specific layers (e.g., output heads) with customizable colors and styles
- **Custom Neuron Labels**: Add text or LaTeX math labels to individual neurons
- **Background Colors**: Transparent (default), white, or any custom color backgrounds
- **Adjustable Figure Size**: Control width and height independently for perfect layouts
- **Smart Layer Collapsing**: Automatically collapse large layers (>N neurons) with ellipsis notation
- **Multiple Export Formats**: Save as PNG, SVG, or PDF with adjustable DPI
- **Customizable Plots**: Control every aspect of the visualization
- **Layer Management**: Add, remove, and query layers with ease
- **Network Analysis**: Check network topology, find root/leaf layers, and analyze connections
- **Auto-Generated IDs**: Each layer gets a unique identifier automatically

## Installation

Clone the repository:
```bash
git clone https://github.com/Matteodambr/NN_PLOT.git
cd NN_PLOT
```

Install required dependencies:
```bash
pip install matplotlib
```

## Quick Start

### Basic Visualization

```python
from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer
from src.NN_PLOTTING_UTILITIES import plot_network

# Create a simple network
nn = NeuralNetwork("My Network")
nn.add_layer(FullyConnectedLayer(num_neurons=4, name="Input"))
nn.add_layer(FullyConnectedLayer(num_neurons=6, activation="relu", name="Hidden"))
nn.add_layer(FullyConnectedLayer(num_neurons=3, activation="softmax", name="Output"))

# Plot it!
plot_network(nn, title="My First Network", save_path="network.png")
```

### Custom Layer Styling

```python
from src.NN_PLOTTING_UTILITIES import PlotConfig, LayerStyle

# Define custom styles for each layer
layer_styles = {
    "Input": LayerStyle(
        neuron_fill_color="lightcoral",
        neuron_edge_color="darkred",
        neuron_edge_width=2.0,
        connection_linewidth=1.5,
        connection_color="red",
        connection_alpha=0.6
    ),
    "Hidden": LayerStyle(
        neuron_fill_color="lightgreen",
        neuron_edge_color="darkgreen",
        neuron_edge_width=2.5,
        connection_linewidth=1.0,
        connection_color="green"
    ),
    "Output": LayerStyle(
        neuron_fill_color="lightblue",
        neuron_edge_color="darkblue",
        neuron_edge_width=3.0
    )
}

# Create configuration with custom styles
config = PlotConfig(
    figsize=(14, 8),
    layer_styles=layer_styles,
    show_layer_names=True  # or False for a clean view
)

# Plot with custom styling
plot_network(nn, config=config, save_path="styled_network.png")
```

### Custom Neuron Labels (with LaTeX Support)

Add descriptive text labels to individual neurons, perfect for documenting what each input feature or output represents. Supports both plain text and LaTeX mathematical notation.

```python
from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig

# Create a network with labeled inputs and outputs
nn = NeuralNetwork("Credit Risk Model")

# Input layer with descriptive labels on the left
nn.add_layer(FullyConnectedLayer(
    num_neurons=4,
    name="Input",
    neuron_labels=["Age", "Income", "Credit Score", "Debt Ratio"],
    label_position="left"
))

nn.add_layer(FullyConnectedLayer(5, activation="relu", name="Hidden"))

# Output layer with labels on the right
nn.add_layer(FullyConnectedLayer(
    num_neurons=2,
    activation="softmax",
    name="Output",
    neuron_labels=["Approved", "Denied"],
    label_position="right"
))

# Enable label display
config = PlotConfig(show_neuron_text_labels=True)
plot_network(nn, config=config, save_path="labeled_network.png")
```

**LaTeX Math Support:**

```python
# Use LaTeX for mathematical notation
nn = NeuralNetwork("Math Model")

nn.add_layer(FullyConnectedLayer(
    num_neurons=3,
    name="Input",
    neuron_labels=[r"$x_1$", r"$x_2$", r"$x_3$"],  # LaTeX math mode
    label_position="left"
))

nn.add_layer(FullyConnectedLayer(4, activation="relu", name="Hidden"))

nn.add_layer(FullyConnectedLayer(
    num_neurons=2,
    name="Output",
    neuron_labels=[r"$\hat{y}_1$", r"$\hat{y}_2$"],  # Predictions with hat notation
    label_position="right"
))

config = PlotConfig(
    show_neuron_text_labels=True,
    neuron_text_label_fontsize=12  # Adjust font size
)
plot_network(nn, config=config, save_path="latex_labels.png")
```

**Key Features:**
- **Plain Text**: Simple descriptive labels like "Age", "Income", "Approved"
- **LaTeX Math**: Use `r"$...$"` for math notation: `$x_1$`, `$\alpha$`, `$\hat{y}$`, `$\frac{a}{b}$`
- **Positioning**: `label_position="left"` or `"right"` controls label placement
- **Show/Hide**: Use `show_neuron_text_labels` in PlotConfig to toggle display
- **Customization**: Adjust `neuron_text_label_fontsize` and `neuron_text_label_offset`
- **Per-Layer**: Each layer can have its own labels (or none at all)


### Linear (Sequential) Network

```python
from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer

# Create a network
nn = NeuralNetwork(name="My Classifier")

# Add layers sequentially (automatically connected)
nn.add_layer(FullyConnectedLayer(num_neurons=784, name="Input"))
nn.add_layer(FullyConnectedLayer(num_neurons=128, activation="relu", name="Hidden"))
nn.add_layer(FullyConnectedLayer(num_neurons=10, activation="softmax", name="Output"))

# Display the network
print(nn)
```

### Branching Network

```python
from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer

# Create a network with branches
nn = NeuralNetwork(name="Multi-Branch Network")

# Add input and shared hidden layer
nn.add_layer(FullyConnectedLayer(num_neurons=100, name="Input"))
input_id = nn.get_layer_id_by_name("Input")

nn.add_layer(FullyConnectedLayer(num_neurons=64, activation="relu", name="Shared"))
shared_id = nn.get_layer_id_by_name("Shared")

# Create parallel branches
nn.add_layer(
    FullyConnectedLayer(num_neurons=32, activation="relu", name="Branch1"),
    parent_ids=[shared_id]
)
branch1_id = nn.get_layer_id_by_name("Branch1")

nn.add_layer(
    FullyConnectedLayer(num_neurons=32, activation="relu", name="Branch2"),
    parent_ids=[shared_id]
)
branch2_id = nn.get_layer_id_by_name("Branch2")

# Merge branches
nn.add_layer(
    FullyConnectedLayer(num_neurons=10, activation="softmax", name="Output"),
    parent_ids=[branch1_id, branch2_id]
)
```

## Visualization Options

### PlotConfig Parameters

- `figsize`: Figure size as (width, height) tuple
- `neuron_radius`: Radius of neuron circles
- `layer_spacing`: Horizontal spacing between layers
- `neuron_spacing`: Vertical spacing between neurons
- `connection_alpha`: Transparency of connection lines (0-1)
- `connection_color`: Default color for connections
- `connection_linewidth`: Default width for connection lines
- `neuron_color`: Default fill color for neurons
- `neuron_edge_color`: Default edge color for neurons
- `neuron_edge_width`: Default width for neuron edges
- `show_neuron_labels`: Whether to show neuron indices/numbers
- `neuron_numbering_reversed`: Whether to reverse numbering direction (N-1 to 0 instead of 0 to N-1)
- `show_neuron_text_labels`: Whether to show custom text labels from layer.neuron_labels (default: True)
- `neuron_text_label_fontsize`: Font size for custom neuron text labels (default: 10)
- `neuron_text_label_offset`: Horizontal offset from neuron center for text labels (default: 0.8)
- `show_layer_names`: Whether to show layer names
- `title_fontsize`: Font size for plot title
- `layer_name_fontsize`: Font size for layer names
- `max_neurons_per_layer`: Maximum neurons to show before collapsing (default: 20)
- `collapse_neurons_start`: Number of neurons to show at start of collapsed layers (default: 10)
- `collapse_neurons_end`: Number of neurons to show at end of collapsed layers (default: 9)
- `layer_styles`: Dictionary mapping layer names/IDs to LayerStyle objects

### LayerStyle Parameters

Each layer can have custom styling:

- `neuron_fill_color`: Fill color for this layer's neurons
- `neuron_edge_color`: Edge color for this layer's neurons
- `neuron_edge_width`: Width of neuron edges
- `connection_linewidth`: Width of connections FROM this layer
- `connection_color`: Color of connections FROM this layer
- `connection_alpha`: Transparency of connections FROM this layer
- `box_around_layer`: If True, draw a rounded box around this layer
- `box_fill_color`: Fill color for the box (use None for no fill, just border)
- `box_edge_color`: Edge color for the box
- `box_edge_width`: Width of the box edge
- `box_padding`: Padding around neurons inside the box
- `box_corner_radius`: Corner radius for the rounded box

### Layer Boxes (New Feature!)

You can now draw rounded boxes around specific layers to highlight them (e.g., output heads):

```python
from src.NN_PLOTTING_UTILITIES import PlotConfig, LayerStyle

# Create network with branching output heads
nn = NeuralNetwork("Boxed Heads")
input_id = nn.add_layer(FullyConnectedLayer(6, name="Input"))
hidden_id = nn.add_layer(FullyConnectedLayer(300, name="Hidden"))

# Two output heads
nn.add_layer(FullyConnectedLayer(7, name="Head_A"), parent_ids=[hidden_id])
nn.add_layer(FullyConnectedLayer(7, name="Head_B"), parent_ids=[hidden_id])

# Configure with boxes around output heads
config = PlotConfig(
    figsize=(16, 8),  # Wider figure for better visibility
    layer_styles={
        'Head_A': LayerStyle(
            neuron_fill_color='lightcoral',
            box_around_layer=True,
            box_fill_color='#FFE6E6',  # Light red fill
            box_edge_color='darkred',
            box_edge_width=2.5,
            box_padding=0.7,
            box_corner_radius=0.4
        ),
        'Head_B': LayerStyle(
            neuron_fill_color='lightgreen',
            box_around_layer=True,
            box_fill_color='#E6FFE6',  # Light green fill
            box_edge_color='darkgreen',
            box_edge_width=2.5,
            box_padding=0.7,
            box_corner_radius=0.4
        )
    }
)

plot_network(nn, config=config, save_path="boxed_heads.png")
```

**Box Style Options:**
- **Filled boxes**: Set `box_fill_color` to a color
- **Border-only boxes**: Set `box_fill_color=None`
- **Adjustable corners**: Change `box_corner_radius` for sharper/rounder corners
- **Custom padding**: Adjust `box_padding` for tighter/looser boxes
- **Text labels**: Automatically positioned outside boxes when present

**Note:** Text labels on neurons are automatically positioned outside the box boundaries!

See `examples/demo_layer_boxes.py` for more examples!

### Network Spacing Control

Control the overall width of your network visualization using the `layer_spacing_multiplier` parameter:

```python
# Default spacing (1.0x)
config_default = PlotConfig(
    layer_spacing_multiplier=1.0  # Default
)

# Wider network (1.5x spacing = 50% wider)
config_wide = PlotConfig(
    layer_spacing_multiplier=1.5  # 50% wider
)

# Much wider network (2.0x spacing = 100% wider)
config_very_wide = PlotConfig(
    layer_spacing_multiplier=2.0  # Double width
)

plot_network(nn, config=config_wide, save_path="wide_network.png")
```

**Use cases:**
- Cramped networks with many layers → use multiplier > 1.0
- Networks with text labels that overlap → increase spacing
- Publication figures that need more horizontal space
- Maintains relative spacing while scaling overall width

See `examples/demo_spacing_and_labels.py` for demonstrations!

### Collapsed Layers

For layers with many neurons (e.g., 100+ neurons), the visualization automatically collapses the layer to keep the plot readable:

```python
# Create a network with a very large layer
nn = NeuralNetwork("Large Network")
nn.add_layer(FullyConnectedLayer(100, name="Input"))  # Large layer
nn.add_layer(FullyConnectedLayer(50, activation="relu", name="Hidden"))
nn.add_layer(FullyConnectedLayer(10, activation="softmax", name="Output"))

# Control the collapse threshold and distribution
config = PlotConfig(
    max_neurons_per_layer=20,      # Layers with >20 neurons will be collapsed
    collapse_neurons_start=10,      # Show 10 neurons at the start
    collapse_neurons_end=9,         # Show 9 neurons at the end
    show_neuron_labels=True         # Show neuron indices (optional)
)

plot_network(nn, config=config, save_path="large_network.png")
```

**How it works:**
- Layers exceeding `max_neurons_per_layer` are collapsed
- Shows first N neurons (configurable via `collapse_neurons_start`)
- Three vertical dots (`⋮`) indicate omitted neurons
- Shows last M neurons (configurable via `collapse_neurons_end`)
- Connections skip the ellipsis position
- Neuron labels (if enabled) show actual indices, not display indices
- Default: 20 max neurons, 10 at start, 9 at end

**Example:** A layer with 100 neurons using defaults displays:
- Neurons 0-9 (first 10)
- Three dots (indicating omitted neurons 10-89)
- Neurons 90-99 (last 10)

**Custom distributions:**
```python
# More at start (good for input layers)
config = PlotConfig(collapse_neurons_start=15, collapse_neurons_end=4)

# More at end (good for output layers)  
config = PlotConfig(collapse_neurons_start=5, collapse_neurons_end=14)

# Minimal (for very large layers)
config = PlotConfig(collapse_neurons_start=3, collapse_neurons_end=3)

# Symmetric
config = PlotConfig(collapse_neurons_start=7, collapse_neurons_end=7)
```

### Neuron Numbering

Control whether to show index numbers on each neuron and their direction:

```python
# Show neuron indices (useful for debugging or documentation)
config = PlotConfig(show_neuron_labels=True)

# Hide neuron indices (cleaner look)
config = PlotConfig(show_neuron_labels=False)  # Default

# Reverse numbering direction (highest index at top)
config = PlotConfig(
    show_neuron_labels=True,
    neuron_numbering_reversed=True
)
```

**Numbering Direction:**
- `neuron_numbering_reversed=False` (default): Neurons numbered 0 to N-1 from top to bottom
- `neuron_numbering_reversed=True`: Neurons numbered N-1 to 0 from top to bottom

This is useful when you want the highest-index neurons at the top of the visualization, or when matching a specific mathematical convention.

### Export Options

Control image quality and format when saving plots:

```python
# High DPI for publication-quality images
plot_network(nn, save_path="network.png", dpi=600)

# SVG format for scalable vector graphics (ideal for presentations)
plot_network(nn, save_path="network.svg", format="svg")

# PDF format
plot_network(nn, save_path="network.pdf", format="pdf")

# Auto-detect format from file extension
plot_network(nn, save_path="network.svg")  # Automatically uses SVG format

# Low DPI for quick previews
plot_network(nn, save_path="preview.png", dpi=72)
```

**DPI Options:**
- Default: 300 DPI (good for most uses)
- 72 DPI: Screen resolution, small file size
- 150 DPI: Good for web/documents
- 300 DPI: Standard print quality
- 600 DPI: High-quality prints and publications

**Format Options:**
- PNG: Raster format, good for most uses (default)
- SVG: Vector format, perfect for presentations and papers (infinitely scalable)
- PDF: Vector format, good for documents
- Format is auto-detected from file extension if not specified

### Background Colors

Control the background color of your visualizations. Default is **transparent**.

```python
from src.NN_PLOTTING_UTILITIES import PlotConfig

# Transparent background (default) - perfect for presentations
config_transparent = PlotConfig(background_color='transparent')
plot_network(nn, config=config_transparent, save_path="transparent.png")

# White background
config_white = PlotConfig(background_color='white')
plot_network(nn, config=config_white, save_path="white.png")

# Custom colors (named colors, hex, or RGB)
config_custom = PlotConfig(background_color='#E6F2FF')  # Light blue
plot_network(nn, config=config_custom, save_path="custom.png")

# Dark mode
config_dark = PlotConfig(
    background_color='#2B2B2B',
    neuron_color='lightblue',
    neuron_edge_color='white',
    connection_color='lightgray'
)
plot_network(nn, config=config_dark, save_path="dark.png")
```

**Background Options:**
- `'transparent'` (default): No background, transparent PNG/SVG
- `'white'`: White background
- Any named color: `'lightgray'`, `'lightblue'`, `'cream'`, etc.
- Hex colors: `'#E6F2FF'`, `'#2B2B2B'`, etc.
- RGB tuples: `(0.9, 0.9, 0.9)` for light gray

### Title Control

Control whether to show the plot title. Default is **True** (title shown).

```python
from src.NN_PLOTTING_UTILITIES import PlotConfig

# With title (default)
config_with = PlotConfig(show_title=True)
plot_network(nn, config=config_with, title="My Network", save_path="with_title.png")

# Without title (clean plot)
config_without = PlotConfig(show_title=False)
plot_network(nn, config=config_without, save_path="no_title.png")
```

**Title Behavior:**
- When `show_title=True` and you provide a title: Uses your custom title
- When `show_title=True` and no title provided: Shows default `"Neural Network: {network.name}"`
- When `show_title=False`: No title shown regardless of title parameter

### Layer Variable Names

Add descriptive labels to input and output layers showing what variables they represent. Perfect for documentation and presentations.

```python
from src.NN_PLOTTING_UTILITIES import PlotConfig

# Create a network
nn = NeuralNetwork("Policy Network")
input_layer = FullyConnectedLayer(num_neurons=6, name="Input")
nn.add_layer(input_layer)
hidden = FullyConnectedLayer(num_neurons=100, activation="relu", name="Hidden")
nn.add_layer(hidden, parent_ids=[input_layer.layer_id])
output = FullyConnectedLayer(num_neurons=3, activation="softmax", name="Output")
nn.add_layer(output, parent_ids=[hidden.layer_id])

# Add variable names to layers
config = PlotConfig(
    layer_variable_names={
        'Input': r'$x_1, x_2, x_3, x_4, x_5, x_6$ (State)',
        'Output': 'Action: Left, Right, Jump'
    },
    show_layer_variable_names=True,        # Enable the feature
    layer_variable_names_fontsize=11,      # Font size
    layer_variable_names_position='side'   # Position: 'side', 'above', or 'below'
)

plot_network(nn, config=config, save_path="network_with_variables.png")
```

**Position Options:**

1. **`'side'` (default)**: Smart positioning
   - Input layers: Labels on the left
   - Output layers: Labels on the right
   - Hidden layers: Labels above

2. **`'above'`**: All labels above layers

3. **`'below'`**: All labels below layers

**LaTeX Support:**
```python
# Use LaTeX mathematical notation
layer_variable_names={
    'Input': r'$\mathbf{x} = [x_1, x_2, x_3]^T$',
    'Output': r'$\hat{y} \in \mathbb{R}^n$'
}
```

**Using Layer IDs:**
```python
# You can also use layer IDs instead of names
input_id = nn.add_layer(FullyConnectedLayer(num_neurons=5))
output_id = nn.add_layer(...)

config = PlotConfig(
    layer_variable_names={
        input_id: 'Sensor Data',
        output_id: 'Control Signal'
    }
)
```

**With Colored Boxes:**
```python
# Variable names automatically adjust for boxes
config = PlotConfig(
    layer_variable_names={
        'Input': 'Features: x, y, z',
        'Output': 'Predictions'
    },
    layer_styles={
        'Input': LayerStyle(
            neuron_fill_color='#FFD700',
            box_around_layer=True,
            box_fill_color='#FFFACD'
        )
    }
)
# Labels automatically position outside the box!
```

**Demo:** Run `python examples/demo_variable_names.py` for 6 complete examples.


## API Reference

### Main Classes

#### `FullyConnectedLayer`

Represents a fully connected (dense) layer.

**Attributes:**
- `num_neurons` (int): Number of neurons in the layer
- `activation` (Optional[str]): Activation function name
- `name` (Optional[str]): Human-readable layer name
- `use_bias` (bool): Whether to use bias terms (default: True)
- `neuron_labels` (Optional[List[str]]): Custom text labels for each neuron (supports LaTeX)
- `label_position` (str): Position of labels relative to neurons ('left' or 'right', default: 'left')

#### `NeuralNetwork`

Main class for representing neural network structure.

**Methods:**
- `add_layer(layer, parent_ids=None)`: Add a layer to the network
- `get_layer(layer_id)`: Get layer by ID
- `get_layer_id_by_name(name)`: Get layer ID by name
- `get_parents(layer_id)`: Get parent layers
- `get_children(layer_id)`: Get child layers
- `is_linear()`: Check if network has sequential structure

#### `PlotConfig`

Configuration for plot customization (see Visualization Options above).

#### `LayerStyle`

Style configuration for individual layers (see Visualization Options above).

### Functions

#### `plot_network(network, title=None, save_path=None, show=True, config=None, dpi=300, format=None)`

Main function to plot a neural network.

**Parameters:**
- `network`: NeuralNetwork object to visualize
- `title`: Optional plot title
- `save_path`: Optional path to save the figure
- `show`: Whether to display the plot (default: True)
- `config`: Optional PlotConfig for customization
- `dpi`: DPI resolution for saved images (default: 300)
- `format`: File format (png, svg, pdf, etc.). Auto-detected from file extension if not specified.

**Returns:** matplotlib Figure object

## Examples

### ⭐ Comprehensive Showcase - START HERE!

**`examples/comprehensive_showcase.py`** - A complete demonstration of ALL features in one script:
- 11 examples covering every feature
- 17 output files (PNG, SVG, PDF formats)
- Fully commented and organized
- Realistic use case examples

Run it to see everything:
```bash
python examples/comprehensive_showcase.py
```

This creates an `outputs/` directory with demonstrations of:
1. Basic network plotting
2. Custom layer styling
3. Plain text neuron labels
4. LaTeX mathematical notation
5. Bold math and complex LaTeX
6. Normal vs reversed neuron numbering
7. Layer collapsing for large networks
8. Different export formats and DPI settings
9. Font customization (Times, Arial, etc.)
10. Show/hide element controls
11. Complete realistic example (Customer Churn Prediction)

### Quick Demos

See the `examples/` directory for quick demonstration scripts:
- `demo_labels.py`: Custom text labels with LaTeX support
- `demo_numbering.py`: Neuron numbering direction options
- `demo_fonts.py`: Different font styles and customization
- `demo_layer_boxes.py`: Rounded boxes around layers (output heads, etc.)
- `demo_spacing_and_labels.py`: Text labels outside boxes & spacing multiplier
- `demo_variable_names.py`: Layer variable names for inputs/outputs (NEW!)

Run any demo:
```bash
python examples/demo_labels.py
python examples/demo_numbering.py
python examples/demo_fonts.py
python examples/demo_layer_boxes.py
python examples/demo_spacing_and_labels.py
python examples/demo_variable_names.py
```

### Comprehensive Tests

See the `src/tests/` directory for comprehensive test suites:
- `test_plot_simple.py`: Basic network plotting
- `test_layer_styles.py`: Layer-specific styling
- `test_hide_names.py`: Showing/hiding layer names
- `test_collapsed_layers.py`: Large layers with automatic collapsing
- `test_neuron_controls.py`: Neuron numbering and collapse distribution
- `test_numbering_and_export.py`: Numbering direction and export format options
- `test_neuron_labels.py`: Custom text labels with LaTeX support
- `test_fonts.py`: Font customization options
- `test_bold_math.py`: Bold mathematical notation

Run any test:
```bash
python src/tests/test_plot_simple.py
python src/tests/test_neuron_labels.py
python src/tests/test_fonts.py
```

### Paper-Specific Figures

The `PlottedNetworks/` directory contains scripts for generating publication-quality figures for specific papers:

- **CEAS2025/**: Scripts for CEAS 2025 conference paper
  - High-resolution PDF/PNG exports optimized for publication
  - White backgrounds and clean layouts for papers
  - Customizable for your specific architecture

To generate figures for a paper:
```bash
python PlottedNetworks/CEAS2025/generate_figures.py
```

See `PlottedNetworks/README.md` for details on adding new paper directories.

## Current Status

✅ Neural network structure representation  
✅ Parent-child layer relationships  
✅ Linear and branching topologies  
✅ Beautiful matplotlib visualizations  
✅ Layer-specific styling  
✅ Customizable plots  
✅ Smart layer collapsing for large networks  
✅ Neuron numbering with reversible direction  
✅ Custom neuron labels with LaTeX support  
✅ Export to various formats (PNG, SVG, PDF) with DPI control  
⬜ Support for more layer types (coming soon)

## License

See LICENSE file for details.




