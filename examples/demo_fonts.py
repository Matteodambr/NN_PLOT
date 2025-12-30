"""
Quick demo showing neural networks with different font styles.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib.pyplot as plt
from NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer, VectorInput
from NN_PLOTTING_UTILITIES import plot_network, PlotConfig

os.makedirs("test_outputs", exist_ok=True)

def create_network():
    """Create a sample network with labels."""
    nn = NeuralNetwork("Sample Network")
    nn.add_layer(VectorInput(
        num_features=3,
        name="Input",
        neuron_labels=[r"$x_1$", r"$x_2$", r"$x_3$"],
        label_position="left"
    ))
    nn.add_layer(FullyConnectedLayer(4, activation="relu", name="Hidden"))
    nn.add_layer(FullyConnectedLayer(
        num_neurons=2,
        name="Output",
        neuron_labels=[r"$\hat{y}_1$", r"$\hat{y}_2$"],
        label_position="right"
    ))
    return nn

config = PlotConfig(show_neuron_text_labels=True, neuron_text_label_fontsize=12)

print("Creating font comparison examples...\n")

# 1. Default (DejaVu Sans)
print("1. Default font (DejaVu Sans)...")
nn = create_network()
plot_network(nn, config=config, title="Default Font: DejaVu Sans", 
             save_path="test_outputs/font_demo_default.png", show=False, dpi=300)
print("   ✓ font_demo_default.png")

# Reset and set Times New Roman
plt.rcParams.update(plt.rcParamsDefault)

# 2. Times New Roman
print("2. Times New Roman (serif)...")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
nn = create_network()
plot_network(nn, config=config, title="Times New Roman Font", 
             save_path="test_outputs/font_demo_times.png", show=False, dpi=300)
print("   ✓ font_demo_times.png")

# Reset
plt.rcParams.update(plt.rcParamsDefault)

# 3. Arial
print("3. Arial (sans-serif)...")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
nn = create_network()
plot_network(nn, config=config, title="Arial Font", 
             save_path="test_outputs/font_demo_arial.png", show=False, dpi=300)
print("   ✓ font_demo_arial.png")

# Reset
plt.rcParams.update(plt.rcParamsDefault)

# 4. Calibri (modern)
print("4. Calibri (modern sans-serif)...")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri', 'DejaVu Sans']
nn = create_network()
plot_network(nn, config=config, title="Calibri Font", 
             save_path="test_outputs/font_demo_calibri.png", show=False, dpi=300)
print("   ✓ font_demo_calibri.png")

# Reset to default
plt.rcParams.update(plt.rcParamsDefault)

print("\n" + "="*60)
print("Font comparison demo complete!")
print("Check test_outputs/ for:")
print("  - font_demo_default.png")
print("  - font_demo_times.png")
print("  - font_demo_arial.png")
print("  - font_demo_calibri.png")
print("="*60)
