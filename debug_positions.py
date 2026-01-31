#!/usr/bin/env python3
"""Debug script to understand the positioning of multi-input ImageInput layers."""

import sys
sys.path.insert(0, 'src')

from NN_DEFINITION_UTILITIES import NeuralNetwork, ImageInput, FullyConnectedLayer, VectorOutput
from NN_PLOTTING_UTILITIES import NetworkPlotter
import os

# Get path to catto.jpg
catto_path = os.path.join('src', 'readme_image_static', 'catto.jpg')

# Create network with 3 inputs of different sizes
network = NeuralNetwork("Multi-Input Debug")

# Small input
small_input = ImageInput(
    height=64, width=64, channels=3,
    name="Small Input",
    display_mode="image",
    image_path=catto_path,
    color_mode="rgb",
    custom_size=1.5
)

# Medium input  
medium_input = ImageInput(
    height=128, width=128, channels=3,
    name="Medium Input",
    display_mode="image",
    image_path=catto_path,
    color_mode="rgb",
    custom_size=2.5
)

# Large input
large_input = ImageInput(
    height=224, width=224, channels=3,
    name="Large Input",
    display_mode="image",
    image_path=catto_path,
    color_mode="rgb",
    custom_size=3.5
)

small_id = network.add_layer(small_input, is_input=True)
medium_id = network.add_layer(medium_input, is_input=True)
large_id = network.add_layer(large_input, is_input=True)

# Merge layer
merge_layer = FullyConnectedLayer(16, name="Merge")
merge_id = network.add_layer(merge_layer, parent_ids=[small_id, medium_id, large_id])

# Output
output = VectorOutput(10, name="Output")
output_id = network.add_layer(output, parent_ids=[merge_id])

# Plot
plotter = NetworkPlotter()
plotter.plot_network(network)

# After plotting, print debug information
print("\n" + "="*80)
print("DEBUG INFORMATION")
print("="*80)

# Print layer positions
print("\nLayer center positions (x, y):")
for layer_id, pos in plotter.layer_positions.items():
    layer = network.get_layer(layer_id)
    print(f"  {layer.name}: {pos}")

# Print neuron positions for each layer
print("\nNeuron positions by layer:")
for layer_id, positions in plotter.neuron_positions.items():
    layer = network.get_layer(layer_id)
    y_positions = [pos[1] for pos in positions]
    print(f"  {layer.name}:")
    print(f"    Center Y: {plotter.layer_positions[layer_id][1]:.2f}")
    if len(y_positions) == 1:
        print(f"    Single position: Y={y_positions[0]:.2f}")
    else:
        print(f"    Y range: {min(y_positions):.2f} to {max(y_positions):.2f}")
        print(f"    Y span: {max(y_positions) - min(y_positions):.2f}")

# Calculate ImageInput visual bounds
print("\nImageInput visual bounds:")
for layer_id in [small_id, medium_id, large_id]:
    layer = network.get_layer(layer_id)
    center_y = plotter.layer_positions[layer_id][1]
    
    # Calculate visual height
    if layer.custom_size is not None:
        size_factor = layer.custom_size
    else:
        aspect_ratio = layer.width / layer.height if layer.height > 0 else 1.0
        size_factor = plotter.config.neuron_radius * 15 / aspect_ratio
    
    if layer.separate_channels:
        offset = size_factor * 0.15
        total_height = size_factor + 2 * offset
    else:
        total_height = size_factor
    
    top = center_y + total_height / 2
    bottom = center_y - total_height / 2
    
    print(f"  {layer.name}:")
    print(f"    Center Y: {center_y:.2f}")
    print(f"    Height: {total_height:.2f}")
    print(f"    Top: {top:.2f}")
    print(f"    Bottom: {bottom:.2f}")

# Calculate overall network bounds
print("\nOverall network bounds:")
all_tops = []
all_bottoms = []

for layer_id in network.layers.keys():
    layer = network.get_layer(layer_id)
    
    if isinstance(layer, ImageInput):
        center_y = plotter.layer_positions[layer_id][1]
        
        if layer.custom_size is not None:
            size_factor = layer.custom_size
        else:
            aspect_ratio = layer.width / layer.height if layer.height > 0 else 1.0
            size_factor = plotter.config.neuron_radius * 15 / aspect_ratio
        
        if layer.separate_channels:
            offset = size_factor * 0.15
            total_height = size_factor + 2 * offset
        else:
            total_height = size_factor
        
        top = center_y + total_height / 2
        bottom = center_y - total_height / 2
        all_tops.append(top)
        all_bottoms.append(bottom)
    else:
        if layer_id in plotter.neuron_positions:
            y_positions = [pos[1] for pos in plotter.neuron_positions[layer_id]]
            if y_positions:
                all_tops.append(max(y_positions))
                all_bottoms.append(min(y_positions))

if all_tops and all_bottoms:
    global_top = max(all_tops)
    global_bottom = min(all_bottoms)
    global_center = (global_top + global_bottom) / 2
    
    print(f"  Global top: {global_top:.2f}")
    print(f"  Global bottom: {global_bottom:.2f}")
    print(f"  Global center: {global_center:.2f}")
    print(f"  Global span: {global_top - global_bottom:.2f}")
    
    if abs(global_center) > 0.01:
        print(f"\n  ⚠️  WARNING: Global center is NOT at y=0! It's at y={global_center:.2f}")
        print(f"  ⚠️  Network is shifted by {global_center:.2f} units")
    else:
        print(f"\n  ✓  Global center is correctly at y=0")

print("="*80)
