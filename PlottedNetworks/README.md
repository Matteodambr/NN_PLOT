# PlottedNetworks

This directory contains paper-specific neural network visualization scripts.

Each subdirectory corresponds to a specific paper/conference and contains scripts
that generate the network architecture figures for that publication.

## Structure

- **CEAS2025/** - Scripts for CEAS 2025 conference paper

## Usage

Each paper directory should contain:
- Python scripts to generate specific network architectures
- Output subdirectory for generated figures (optional)
- README with paper details and figure descriptions (optional)

## Adding a New Paper

1. Create a new directory: `PlottedNetworks/PaperName/`
2. Add your network generation scripts
3. Use the NN_PLOT library to create publication-quality figures

Example:
```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig

# Create your network
nn = NeuralNetwork("Paper Architecture")
nn.add_layer(FullyConnectedLayer(10, name="Input"))
# ... add more layers

# Configure for publication
config = PlotConfig(
    background_color='white',  # or 'transparent'
    show_title=False,  # Clean plot for papers
    figsize=(10, 6)
)

# Generate high-quality figure
plot_network(nn, config=config, save_path="figure1.pdf", dpi=600, format="pdf")
```
