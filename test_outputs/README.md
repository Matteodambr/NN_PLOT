# Test Outputs

This directory contains generated plots from test scripts.

## Prerequisites

Before running the tests, ensure you have matplotlib installed:

```bash
pip install matplotlib
```

## Generating Plots

The plots are generated dynamically when you run the test scripts. They are **not** committed to the repository as they are large binary files.

To generate the output layer demonstration plots, run:

```bash
python3 src/tests/test_output_layers.py
```

This will create the following plots in this directory:
- `output_layer_classification.png` - VectorOutput with individual neurons for classification
- `output_layer_regression.png` - GenericOutput rounded box for regression
- `output_layer_multi_output.png` - Multi-output network with both VectorOutput and GenericOutput
- `output_layer_custom_text.png` - GenericOutput with custom text
- `output_layer_styled.png` - VectorOutput with custom styling

## Other Test Outputs

Many other test scripts in `src/tests/` also generate plots in this directory. Run any test file to see the output:

```bash
python3 src/tests/test_plot_simple.py
python3 src/tests/test_bold_math.py
# ... etc
```

## Note

All `.png`, `.svg`, `.jpg`, `.jpeg`, and `.pdf` files are gitignored to keep the repository size small. You need to run the tests locally to generate these files.
