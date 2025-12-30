"""
Test script to explore font support in matplotlib for neuron labels.
Shows available fonts and how to customize them.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# List available fonts
print("=" * 60)
print("AVAILABLE FONTS IN MATPLOTLIB")
print("=" * 60)

# Get all available fonts
fonts = sorted(set([f.name for f in fm.fontManager.ttflist]))

print(f"\nTotal fonts available: {len(fonts)}")
print("\nCommon fonts for technical documents:")
common_fonts = ['DejaVu Sans', 'DejaVu Serif', 'Arial', 'Times New Roman', 
                'Courier New', 'Helvetica', 'Computer Modern']

for font in common_fonts:
    if font in fonts:
        print(f"  ✓ {font}")
    else:
        print(f"  ✗ {font} (not available)")

print("\nAll available fonts:")
for i, font in enumerate(fonts, 1):
    print(f"  {i:3d}. {font}")

print("\n" + "=" * 60)
print("FONT CONFIGURATION OPTIONS")
print("=" * 60)

print("""
Matplotlib supports these font families:
  - 'serif': Times-like fonts (e.g., Times New Roman, DejaVu Serif)
  - 'sans-serif': Helvetica-like fonts (e.g., Arial, DejaVu Sans)
  - 'monospace': Fixed-width fonts (e.g., Courier)
  - 'cursive': Script-style fonts
  - 'fantasy': Decorative fonts

Default matplotlib font settings:
  - Math text: Computer Modern (LaTeX default)
  - Regular text: DejaVu Sans
  - Math font family: 'dejavusans' or 'cm' (Computer Modern)
""")

print("\n" + "=" * 60)
print("TESTING FONTS WITH NEURAL NETWORK LABELS")
print("=" * 60)

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer, VectorInput
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig

# Create output directory at project root
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
output_dir = os.path.join(project_root, "test_outputs", "fonts")
os.makedirs(output_dir, exist_ok=True)

# Test 1: Default font
print("\n1. Testing with default font...")
nn = NeuralNetwork("Font Test")
nn.add_layer(VectorInput(
    num_features=3,
    name="Input",
    neuron_labels=["Feature A", "Feature B", "Feature C"],
    label_position="left"
))
nn.add_layer(FullyConnectedLayer(3, activation="relu", name="Hidden"))
nn.add_layer(FullyConnectedLayer(
    num_neurons=2,
    name="Output",
    neuron_labels=[r"$\hat{y}_1$", r"$\hat{y}_2$"],
    label_position="right"
))

config = PlotConfig(show_neuron_text_labels=True)
fig = plot_network(nn, config=config, title="Default Font", show=False)
fig.savefig("test_outputs/fonts/01_default_font.png", dpi=300)
plt.close(fig)
print("   ✓ Created: 01_default_font.png")

# Test 2: Using rcParams to change font globally
print("\n2. Testing with Times New Roman (serif)...")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']

fig = plot_network(nn, config=config, title="Times New Roman Font", show=False)
fig.savefig("test_outputs/fonts/02_times_font.png", dpi=300)
plt.close(fig)
print("   ✓ Created: 02_times_font.png")

# Reset to default
plt.rcParams.update(plt.rcParamsDefault)

# Test 3: Sans-serif
print("\n3. Testing with Arial/Helvetica (sans-serif)...")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']

fig = plot_network(nn, config=config, title="Arial/Helvetica Font", show=False)
fig.savefig("test_outputs/fonts/03_arial_font.png", dpi=300)
plt.close(fig)
print("   ✓ Created: 03_arial_font.png")

# Reset to default
plt.rcParams.update(plt.rcParamsDefault)

# Test 4: Different font sizes
print("\n4. Testing different font sizes...")
sizes = [8, 10, 14, 18]
for size in sizes:
    config_size = PlotConfig(
        show_neuron_text_labels=True,
        neuron_text_label_fontsize=size
    )
    fig = plot_network(nn, config=config_size, title=f"Font Size {size}pt", show=False)
    fig.savefig(f"test_outputs/fonts/04_fontsize_{size}pt.png", dpi=300)
    plt.close(fig)
    print(f"   ✓ Created: 04_fontsize_{size}pt.png")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Font Customization Options:

1. GLOBAL FONT CHANGE (affects all matplotlib plots):
   
   import matplotlib.pyplot as plt
   
   # Set to serif (Times-like)
   plt.rcParams['font.family'] = 'serif'
   plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
   
   # Set to sans-serif (Helvetica-like)
   plt.rcParams['font.family'] = 'sans-serif'
   plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
   
   # Then create your plot
   plot_network(nn, ...)

2. FONT SIZE (in PlotConfig):
   
   config = PlotConfig(
       neuron_text_label_fontsize=12,  # Custom text labels
       title_fontsize=16,               # Plot title
       layer_name_fontsize=12           # Layer names
   )

3. LaTeX MATH FONTS:
   
   # Default: Computer Modern (LaTeX standard)
   # Can be changed with rcParams:
   plt.rcParams['mathtext.fontset'] = 'cm'          # Computer Modern
   plt.rcParams['mathtext.fontset'] = 'stix'        # STIX fonts
   plt.rcParams['mathtext.fontset'] = 'dejavusans'  # DejaVu Sans
   
4. FONT WEIGHT:
   
   # Bold math: Use LaTeX commands in labels
   neuron_labels=[r"$\\mathbf{x}_1$", r"$\\boldsymbol{\\alpha}$"]

All test files saved to: test_outputs/fonts/
""")

print("✅ Font exploration complete!")
