# CEAS 2025 Conference Paper - Network Figures

This directory contains scripts to generate neural network architecture diagrams for the CEAS 2025 conference paper.

## Paper Information

**Conference:** CEAS 2025 (Council of European Aerospace Societies)  
**Paper Title:** [Add your paper title here]  
**Authors:** [Add authors]  
**Paper ID/Track:** [Add if applicable]

## Files

- `generate_figures.py` - Main script to generate all figures for the paper
- `figures/` - Output directory for generated figures

## Usage

Run the script to generate all figures:

```bash
python PlottedNetworks/CEAS2025/generate_figures.py
```

Or from within this directory:

```bash
cd PlottedNetworks/CEAS2025
python generate_figures.py
```

## Generated Figures

### Figure 1: Main Network Architecture
- **File:** `figures/figure1_network_architecture.pdf` (for paper)
- **File:** `figures/figure1_network_architecture.png` (for presentations)
- **Description:** Branching neural network with 6 inputs, three hidden layers (300 neurons each), and two output heads (7 neurons each)
- **Architecture:**
  - Input: 6 neurons
  - Hidden Layer 1: 300 neurons (ReLU)
  - Hidden Layer 2: 300 neurons (ReLU)
  - Hidden Layer 3: 300 neurons (ReLU)
  - Output Head 1: 7 neurons (Softmax)
  - Output Head 2: 7 neurons (Softmax)
- **Appearance:** Clean layout with no layer labels or neuron numbering
- **Format:** PDF at 600 DPI for publication, PNG at 300 DPI for slides
- **Background:** White
- **Note:** Large layers (300 neurons) are automatically collapsed showing first 10 and last 9 neurons

## Customization

Edit `generate_figures.py` to customize:

### Network Architecture
Modify the layer definitions to match your actual network:
```python
nn_main.add_layer(FullyConnectedLayer(
    num_neurons=10,
    name="Input Layer",
    neuron_labels=["Feature 1", "Feature 2", ...],
    label_position="left"
))
```

### Appearance Settings
Adjust `PlotConfig` for your paper's style guide:
```python
config_publication = PlotConfig(
    figsize=(12, 8),              # Figure size
    background_color='white',      # 'white' or 'transparent'
    show_title=False,              # Typically False for papers
    show_layer_names=True,         # Show/hide layer labels
    neuron_text_label_fontsize=10, # Label font size
    # ... more options
)
```

### Export Formats
Generate different formats as needed:
- **PDF:** Best for LaTeX documents (`format="pdf"`)
- **PNG:** Good for PowerPoint/Keynote
- **SVG:** Vector format for editing
- **EPS:** Some journals require EPS format

## LaTeX Integration

To include in your LaTeX document:

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{path/to/figure1_network_architecture.pdf}
    \caption{Neural network architecture with [describe your network]}
    \label{fig:network_architecture}
\end{figure}
```

## Tips for Publication Figures

1. **Resolution:** Use 600 DPI for final publication PDFs
2. **Fonts:** Ensure text is readable at printed size
3. **Colors:** Check if journal requires grayscale or allows color
4. **File Size:** PDFs are typically smaller than high-DPI PNGs
5. **Consistency:** Use the same styling for all figures in the paper

## Notes

- All figures use white background by default (common for papers)
- Title is hidden (add captions in LaTeX instead)
- Layer names and neuron labels are shown
- Network collapses layers with >15 neurons for readability

## Adding More Figures

Uncomment the Figure 2 section in `generate_figures.py` and modify as needed, or duplicate the Figure 1 code block.
