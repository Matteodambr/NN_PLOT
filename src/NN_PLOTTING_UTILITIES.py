"""
NN_PLOTTING_UTILITIES - A module for visualizing neural network architectures.

This module provides functions to plot neural network structures with neurons
represented as circles and connections as lines.
Currently supports visualization of feedforward neural networks with fully connected layers.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import matplotlib as mpl
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field

# Import from the definition utilities
try:
    from .NN_DEFINITION_UTILITIES import (
        NeuralNetwork,
        FullyConnectedLayer
    )
except ImportError:
    from NN_DEFINITION_UTILITIES import (
        NeuralNetwork,
        FullyConnectedLayer
    )


@dataclass
class LayerStyle:
    """
    Style configuration for a specific layer.
    
    Attributes:
        neuron_fill_color: Fill color for neurons in this layer
        neuron_edge_color: Edge color for neurons in this layer
        neuron_edge_width: Width of neuron circle edges
        connection_linewidth: Width of connection lines FROM this layer to its children
        connection_color: Color of connection lines FROM this layer to its children
        connection_alpha: Transparency of connection lines FROM this layer
        box_around_layer: If True, draw a rounded box around this layer
        box_fill_color: Fill color for the box (use None for no fill)
        box_edge_color: Edge color for the box
        box_edge_width: Width of the box edge
        box_padding: Padding around neurons inside the box (in plot units)
        box_corner_radius: Corner radius for the rounded box (in plot units)
    """
    neuron_fill_color: Optional[str] = None
    neuron_edge_color: Optional[str] = None
    neuron_edge_width: Optional[float] = None
    connection_linewidth: Optional[float] = None
    connection_color: Optional[str] = None
    connection_alpha: Optional[float] = None
    box_around_layer: bool = False
    box_fill_color: Optional[str] = None
    box_edge_color: str = 'black'
    box_edge_width: float = 2.0
    box_padding: float = 0.5
    box_corner_radius: float = 0.3


@dataclass
class PlotConfig:
    """
    Configuration for neural network plotting.
    
    Attributes:
        figsize: Tuple of (width, height) for the figure size in inches. 
                 Adjust width to control horizontal space, height for vertical space.
                 Example: (14, 8) for wider plot, (12, 10) for taller plot.
        neuron_radius: Radius of neuron circles
        layer_spacing: Horizontal spacing between layers
        neuron_spacing: Vertical spacing between neurons in a layer
        connection_alpha: Transparency of connection lines (0-1)
        connection_color: Color of connection lines
        connection_linewidth: Width of connection lines
        neuron_color: Default color for neuron circles
        neuron_edge_color: Edge color for neuron circles
        neuron_edge_width: Width of neuron circle edges
        show_neuron_labels: Whether to show neuron indices/numbers on each neuron
        neuron_numbering_reversed: If True, number neurons bottom-to-top; if False, top-to-bottom
        show_neuron_text_labels: Whether to show custom text labels (from layer.neuron_labels)
        neuron_text_label_fontsize: Font size for custom neuron text labels
        neuron_text_label_offset: Horizontal offset from neuron center for text labels
        show_layer_names: Whether to show layer names
        show_title: Whether to show the plot title
        title_fontsize: Font size for plot title
        layer_name_fontsize: Font size for layer names
        max_neurons_per_layer: Maximum neurons to show per layer (for large layers)
        collapse_neurons_start: Number of neurons to show at start when collapsing
        collapse_neurons_end: Number of neurons to show at end when collapsing
        layer_styles: Dictionary mapping layer IDs or names to LayerStyle objects.
                     Use this to apply layer-specific styling including rounded boxes.
        background_color: Background color for the plot. Use 'transparent' for transparent background,
                         'white' for white, or any matplotlib color (hex, rgb, named colors).
                         Default is 'transparent'.
        layer_spacing_multiplier: Multiplier for the overall network width. Values > 1.0 increase
                                 spacing between layers proportionally, making the network wider.
                                 Default is 1.0 (no scaling). Example: 1.5 makes network 50% wider.
        layer_variable_names: Dictionary mapping layer IDs or names to variable name labels.
                             Example: {'Input': 'Input Variables: x, y, z', 'Output_Head_1': 'Actions: a1, a2'}
        show_layer_variable_names: Whether to show the variable name labels for layers.
        layer_variable_names_fontsize: Font size for layer variable name labels.
        layer_variable_names_position: Position for variable names ('above', 'below', or 'side').
                                      'above' places labels above the layer, 'below' places below,
                                      'side' places to the left for input layers and right for output layers.
        font_family: Font family to use for all text in the plot (including math text).
                    Default is 'Times New Roman'. Other options: 'Arial', 'Helvetica', 'DejaVu Sans', etc.
    """
    figsize: Tuple[float, float] = (12, 8)
    neuron_radius: float = 0.3
    layer_spacing: float = 3.0
    neuron_spacing: float = 1.0
    connection_alpha: float = 0.3
    connection_color: str = 'gray'
    connection_linewidth: float = 1.5
    neuron_color: str = 'lightblue'
    neuron_edge_color: str = 'navy'
    neuron_edge_width: float = 1.5
    show_neuron_labels: bool = False
    neuron_numbering_reversed: bool = False
    show_neuron_text_labels: bool = True
    neuron_text_label_fontsize: int = 10
    neuron_text_label_offset: float = 0.8
    show_layer_names: bool = True
    show_title: bool = True
    title_fontsize: int = 16
    layer_name_fontsize: int = 12
    max_neurons_per_layer: int = 20
    collapse_neurons_start: int = 10
    collapse_neurons_end: int = 9
    layer_styles: Dict[str, LayerStyle] = field(default_factory=dict)
    background_color: str = 'transparent'
    layer_spacing_multiplier: float = 1.0
    layer_variable_names: Dict[str, str] = field(default_factory=dict)
    show_layer_variable_names: bool = True
    layer_variable_names_fontsize: int = 11
    layer_variable_names_position: str = 'side'
    font_family: str = 'Times New Roman'


class NetworkPlotter:
    """
    A class for plotting neural network architectures.
    
    This plotter visualizes neural networks with neurons as circles and
    connections between layers as lines.
    """
    
    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize the NetworkPlotter.
        
        Args:
            config: PlotConfig object with plotting parameters. If None, uses defaults.
        """
        self.config = config or PlotConfig()
        # Set font for all text in the plot (including math text)
        try:
            # Use LaTeX for publication-quality math rendering with proper accents
            mpl.rcParams['text.usetex'] = True
            mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'
            mpl.rcParams['font.family'] = 'serif'
            mpl.rcParams['font.serif'] = [self.config.font_family]
        except Exception:
            # If LaTeX rendering fails, fall back to mathtext
            try:
                mpl.rcParams['text.usetex'] = False
                mpl.rcParams['font.family'] = 'serif'
                mpl.rcParams['font.serif'] = [self.config.font_family]
                mpl.rcParams['mathtext.fontset'] = 'custom'
                mpl.rcParams['mathtext.rm'] = self.config.font_family
                mpl.rcParams['mathtext.it'] = f'{self.config.font_family}:italic'
                mpl.rcParams['mathtext.bf'] = f'{self.config.font_family}:bold'
                mpl.rcParams['mathtext.default'] = 'regular'
                mpl.rcParams['mathtext.cal'] = f'{self.config.font_family}'
                mpl.rcParams['mathtext.tt'] = f'{self.config.font_family}'
                mpl.rcParams['mathtext.sf'] = f'{self.config.font_family}'
            except Exception:
                pass
        self.layer_positions: Dict[str, Tuple[float, float]] = {}
        self.neuron_positions: Dict[str, List[Tuple[float, float]]] = {}
        self.collapsed_layers: Dict[str, bool] = {}  # Track which layers are collapsed
        self.collapsed_info: Dict[str, Dict] = {}  # Store info about collapsed neurons
    
    def _get_layer_style(self, layer_id: str, layer_name: Optional[str]) -> LayerStyle:
        """
        Get the layer-specific style, checking both layer_id and layer_name.
        Returns a LayerStyle with None values if no custom style is set.
        
        Args:
            layer_id: The unique ID of the layer
            layer_name: The name of the layer (if any)
            
        Returns:
            LayerStyle object (may have all None values if no custom style)
        """
        # Try to get style by layer_id first, then by layer_name
        if layer_id in self.config.layer_styles:
            return self.config.layer_styles[layer_id]
        elif layer_name and layer_name in self.config.layer_styles:
            return self.config.layer_styles[layer_name]
        else:
            return LayerStyle()  # Return empty style (all None values)
    
    def plot_network(
        self,
        network: NeuralNetwork,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        dpi: int = 300,
        format: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot a neural network structure.
        
        Args:
            network: NeuralNetwork object to visualize
            title: Optional title for the plot. If None, uses network name
            save_path: Optional path to save the figure
            show: Whether to display the plot
            dpi: DPI (dots per inch) for saving the figure (default: 300)
            format: File format ('png', 'svg', 'pdf', etc.). If None, inferred from save_path
            
        Returns:
            matplotlib Figure object
            
        Raises:
            ValueError: If network is empty or has unsupported layer types
        """
        if network.num_layers() == 0:
            raise ValueError("Cannot plot an empty network")
        
        # Check if network is linear or branching
        is_linear = network.is_linear()
        
        if is_linear:
            return self._plot_linear_network(network, title, save_path, show, dpi, format)
        else:
            return self._plot_branching_network(network, title, save_path, show, dpi, format)
    
    def _plot_linear_network(
        self,
        network: NeuralNetwork,
        title: Optional[str],
        save_path: Optional[str],
        show: bool,
        dpi: int = 300,
        format: Optional[str] = None
    ) -> plt.Figure:
        """Plot a linear (sequential) neural network."""
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Set background color
        if self.config.background_color != 'transparent':
            fig.patch.set_facecolor(self.config.background_color)
            ax.set_facecolor(self.config.background_color)
        
        # Calculate positions for all neurons
        self._calculate_linear_positions(network)
        
        # Draw connections first (so they appear behind neurons)
        self._draw_linear_connections(ax, network)
        
        # Draw neurons
        self._draw_neurons(ax, network)
        
        # Draw boxes around layers (if configured)
        self._draw_layer_boxes(ax, network)
        
        # Add layer names
        if self.config.show_layer_names:
            self._add_layer_labels(ax, network)
        
        # Add variable names for layers
        if self.config.show_layer_variable_names and self.config.layer_variable_names:
            self._add_layer_variable_names(ax, network)
        
        # Set title (if enabled)
        if self.config.show_title:
            plot_title = title or f"Neural Network: {network.name}"
            ax.set_title(plot_title, fontsize=self.config.title_fontsize, pad=20)
        
        # Calculate and set axis limits based on neuron positions and boxes
        self._set_axis_limits(ax, network)
        
        # Set axis properties
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            # Determine format from extension if not specified
            save_format = format
            if save_format is None and save_path:
                save_format = save_path.split('.')[-1].lower()
            
            # Set transparency based on background color
            is_transparent = self.config.background_color == 'transparent'
            
            plt.savefig(
                save_path, 
                dpi=dpi, 
                bbox_inches='tight', 
                format=save_format,
                transparent=is_transparent,
                facecolor=fig.get_facecolor() if not is_transparent else 'none'
            )
        
        # Show if requested
        if show:
            plt.show()
        
        return fig
    
    def _plot_branching_network(
        self,
        network: NeuralNetwork,
        title: Optional[str],
        save_path: Optional[str],
        show: bool,
        dpi: int = 300,
        format: Optional[str] = None
    ) -> plt.Figure:
        """Plot a branching (non-linear) neural network."""
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Set background color
        if self.config.background_color != 'transparent':
            fig.patch.set_facecolor(self.config.background_color)
            ax.set_facecolor(self.config.background_color)
        
        # Calculate positions using a layer-based approach
        self._calculate_branching_positions(network)
        
        # Draw connections first
        self._draw_branching_connections(ax, network)
        
        # Draw neurons
        self._draw_neurons(ax, network)
        
        # Draw boxes around layers (if configured)
        self._draw_layer_boxes(ax, network)
        
        # Add layer names
        if self.config.show_layer_names:
            self._add_layer_labels(ax, network)
        
        # Add variable names for layers
        if self.config.show_layer_variable_names and self.config.layer_variable_names:
            self._add_layer_variable_names(ax, network)
        
        # Set title (if enabled)
        if self.config.show_title:
            plot_title = title or f"Neural Network: {network.name} (Branching)"
            ax.set_title(plot_title, fontsize=self.config.title_fontsize, pad=20)
        
        # Calculate and set axis limits based on neuron positions and boxes
        self._set_axis_limits(ax, network)
        
        # Set axis properties
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            # Determine format from extension if not specified
            save_format = format
            if save_format is None and save_path:
                save_format = save_path.split('.')[-1].lower()
            
            # Set transparency based on background color
            is_transparent = self.config.background_color == 'transparent'
            
            plt.savefig(
                save_path, 
                dpi=dpi, 
                bbox_inches='tight', 
                format=save_format,
                transparent=is_transparent,
                facecolor=fig.get_facecolor() if not is_transparent else 'none'
            )
        
        # Show if requested
        if show:
            plt.show()
        
        return fig
    
    def _calculate_linear_positions(self, network: NeuralNetwork) -> None:
        """Calculate positions for neurons in a linear network."""
        self.neuron_positions.clear()
        self.layer_positions.clear()
        self.collapsed_layers.clear()
        self.collapsed_info.clear()
        
        layer_order = network._layer_order
        
        for i, layer_id in enumerate(layer_order):
            layer = network.get_layer(layer_id)
            
            # X position for this layer (apply spacing multiplier)
            x_pos = i * self.config.layer_spacing * self.config.layer_spacing_multiplier
            
            # Get actual number of neurons
            if isinstance(layer, FullyConnectedLayer):
                actual_neurons = layer.num_neurons
            else:
                actual_neurons = layer.get_output_size()
            
            # Check if we need to collapse this layer
            if actual_neurons > self.config.max_neurons_per_layer:
                self.collapsed_layers[layer_id] = True
                # Use configured collapse distribution
                show_start = self.config.collapse_neurons_start
                show_end = self.config.collapse_neurons_end
                
                # Store collapse info
                self.collapsed_info[layer_id] = {
                    'actual_count': actual_neurons,
                    'show_start': show_start,
                    'show_end': show_end,
                    'dots_position': show_start  # Position where dots go
                }
                
                num_neurons_display = show_start + 1 + show_end  # start + dots + end
            else:
                self.collapsed_layers[layer_id] = False
                num_neurons_display = actual_neurons
            
            # Calculate Y positions for neurons (centered)
            total_height = (num_neurons_display - 1) * self.config.neuron_spacing
            y_start = -total_height / 2
            
            positions = []
            for j in range(num_neurons_display):
                y_pos = y_start + j * self.config.neuron_spacing
                positions.append((x_pos, y_pos))
            
            self.neuron_positions[layer_id] = positions
            
            # Store layer center position for labels
            self.layer_positions[layer_id] = (x_pos, y_start + total_height / 2)
    
    def _calculate_branching_positions(self, network: NeuralNetwork) -> None:
        """Calculate positions for neurons in a branching network using level-based layout."""
        self.neuron_positions.clear()
        self.layer_positions.clear()
        self.collapsed_layers.clear()
        self.collapsed_info.clear()
        
        # Perform topological sort to get layers in levels
        levels = self._compute_layer_levels(network)
        
        # First pass: determine display neuron counts and check for collapsing
        layer_display_counts = {}
        for level_idx, layer_ids in enumerate(levels):
            for layer_id in layer_ids:
                layer = network.get_layer(layer_id)
                
                # Get actual number of neurons
                if isinstance(layer, FullyConnectedLayer):
                    actual_neurons = layer.num_neurons
                else:
                    actual_neurons = layer.get_output_size()
                
                # Check if we need to collapse this layer
                if actual_neurons > self.config.max_neurons_per_layer:
                    self.collapsed_layers[layer_id] = True
                    show_start = self.config.collapse_neurons_start
                    show_end = self.config.collapse_neurons_end
                    
                    self.collapsed_info[layer_id] = {
                        'actual_count': actual_neurons,
                        'show_start': show_start,
                        'show_end': show_end,
                        'dots_position': show_start
                    }
                    
                    num_neurons_display = show_start + 1 + show_end
                else:
                    self.collapsed_layers[layer_id] = False
                    num_neurons_display = actual_neurons
                
                layer_display_counts[layer_id] = num_neurons_display
        
        # Calculate positions for each level
        for level_idx, layer_ids in enumerate(levels):
            # X position for this level (apply spacing multiplier)
            x_pos = level_idx * self.config.layer_spacing * self.config.layer_spacing_multiplier
            
            # Calculate total vertical space needed for this level
            num_layers_in_level = len(layer_ids)
            
            # Calculate heights for all layers in this level
            layer_heights = []
            for layer_id in layer_ids:
                num_neurons_display = layer_display_counts[layer_id]
                total_height = (num_neurons_display - 1) * self.config.neuron_spacing
                layer_heights.append(total_height)
            
            # Calculate total required vertical space with padding between layers
            vertical_padding = 3.0  # Extra space between layers at the same level
            total_vertical_space = sum(layer_heights) + (num_layers_in_level - 1) * vertical_padding
            
            # Start from top and work down
            current_y_offset = total_vertical_space / 2
            
            for sub_idx, layer_id in enumerate(layer_ids):
                num_neurons_display = layer_display_counts[layer_id]
                layer_height = layer_heights[sub_idx]
                
                # Center of this layer
                vertical_offset = current_y_offset - layer_height / 2
                
                # Calculate Y positions for neurons
                y_start = vertical_offset - layer_height / 2
                
                positions = []
                for j in range(num_neurons_display):
                    y_pos = y_start + j * self.config.neuron_spacing
                    positions.append((x_pos, y_pos))
                
                self.neuron_positions[layer_id] = positions
                
                # Store layer center position
                self.layer_positions[layer_id] = (x_pos, vertical_offset)
                
                # Move down for next layer
                current_y_offset -= (layer_height + vertical_padding)
    
    def _compute_layer_levels(self, network: NeuralNetwork) -> List[List[str]]:
        """
        Compute the level (depth) of each layer for branching visualization.
        Returns a list of lists, where each inner list contains layer IDs at that level.
        """
        # Find root layers
        root_layers = network.get_root_layers()
        
        # BFS to assign levels
        levels: List[List[str]] = []
        visited: Set[str] = set()
        current_level = root_layers[:]
        
        while current_level:
            levels.append(current_level[:])
            next_level = []
            
            for layer_id in current_level:
                visited.add(layer_id)
                children = network.get_children(layer_id)
                
                for child_id in children:
                    if child_id not in visited and child_id not in next_level:
                        # Check if all parents have been visited
                        parents = network.get_parents(child_id)
                        if all(p in visited for p in parents):
                            next_level.append(child_id)
            
            current_level = next_level
        
        return levels
    
    def _draw_neurons(self, ax: plt.Axes, network: NeuralNetwork) -> None:
        """Draw neurons as circles, with ellipsis for collapsed layers."""
        for layer_id, positions in self.neuron_positions.items():
            layer = network.get_layer(layer_id)
            
            # Get layer-specific style or use defaults
            layer_style = self._get_layer_style(layer_id, layer.name)
            
            # Determine colors and edge properties
            if isinstance(layer, FullyConnectedLayer):
                fill_color = layer_style.neuron_fill_color or self.config.neuron_color
            else:
                fill_color = layer_style.neuron_fill_color or 'lightgreen'
            
            edge_color = layer_style.neuron_edge_color or self.config.neuron_edge_color
            edge_width = layer_style.neuron_edge_width if layer_style.neuron_edge_width is not None else self.config.neuron_edge_width
            
            # Check if this layer is collapsed
            is_collapsed = self.collapsed_layers.get(layer_id, False)
            dots_position = self.collapsed_info.get(layer_id, {}).get('dots_position', -1) if is_collapsed else -1
            
            # Pre-calculate label x-position for vertical alignment (if labels are enabled)
            layer_label_x = None
            if (self.config.show_neuron_text_labels and 
                isinstance(layer, FullyConnectedLayer) and 
                layer.neuron_labels is not None):
                # Get the x-coordinate of the layer (all neurons in same layer have same x)
                if positions:
                    layer_x = positions[0][0]  # All neurons have same x in a layer
                    if layer.label_position == "left":
                        layer_label_x = layer_x - self.config.neuron_text_label_offset
                    else:  # "right"
                        layer_label_x = layer_x + self.config.neuron_text_label_offset
            
            for i, (x, y) in enumerate(positions):
                # Draw ellipsis dots instead of a neuron at the collapse position
                if is_collapsed and i == dots_position:
                    # Draw three dots vertically
                    dot_spacing = self.config.neuron_radius * 0.8
                    for dot_offset in [-dot_spacing, 0, dot_spacing]:
                        dot = mpatches.Circle(
                            (x, y + dot_offset),
                            self.config.neuron_radius * 0.2,
                            facecolor=edge_color,
                            edgecolor=edge_color,
                            linewidth=0,
                            zorder=10
                        )
                        ax.add_patch(dot)
                else:
                    # Draw regular neuron circle
                    circle = mpatches.Circle(
                        (x, y),
                        self.config.neuron_radius,
                        facecolor=fill_color,
                        edgecolor=edge_color,
                        linewidth=edge_width,
                        zorder=10
                    )
                    ax.add_patch(circle)
                
                # Add neuron labels if requested (skip for dots position)
                if self.config.show_neuron_labels and not (is_collapsed and i == dots_position):
                    # Calculate actual neuron index for collapsed layers
                    if is_collapsed:
                        collapse_info = self.collapsed_info[layer_id]
                        if i < dots_position:
                            # First few neurons
                            actual_index = i
                        else:
                            # Last few neurons (after the dots)
                            actual_index = collapse_info['actual_count'] - (len(positions) - i - 1)
                    else:
                        actual_index = i
                    
                    # Apply numbering direction
                    if self.config.neuron_numbering_reversed:
                        # Reverse: bottom-to-top (higher indices at top)
                        if isinstance(layer, FullyConnectedLayer):
                            total_neurons = layer.num_neurons
                        else:
                            total_neurons = layer.get_output_size()
                        actual_index = total_neurons - 1 - actual_index
                    
                    ax.text(
                        x, y, str(actual_index),
                        ha='center', va='center',
                        fontsize=8,
                        zorder=11
                    )
                
                # Add custom text labels if requested (skip for dots position)
                if (self.config.show_neuron_text_labels and 
                    isinstance(layer, FullyConnectedLayer) and 
                    layer.neuron_labels is not None and
                    not (is_collapsed and i == dots_position)):
                    
                    # Calculate actual neuron index for collapsed layers
                    if is_collapsed:
                        collapse_info = self.collapsed_info[layer_id]
                        if i < dots_position:
                            # First few neurons
                            label_index = i
                        else:
                            # Last few neurons (after the dots)
                            label_index = collapse_info['actual_count'] - (len(positions) - i - 1)
                    else:
                        label_index = i
                    
                    # Apply numbering direction to label index
                    if self.config.neuron_numbering_reversed:
                        total_neurons = layer.num_neurons
                        label_index = total_neurons - 1 - label_index
                    
                    # Get the text label for this neuron
                    if 0 <= label_index < len(layer.neuron_labels):
                        label_text = layer.neuron_labels[label_index]
                        
                        # Use pre-calculated label position with center alignment
                        # All labels in the same layer align at the same vertical line,
                        # centered horizontally on that line
                        label_x = layer_label_x
                        
                        # Draw the text label with LaTeX support (always center-aligned)
                        ax.text(
                            label_x, y, label_text,
                            ha='center', va='center',
                            fontsize=self.config.neuron_text_label_fontsize,
                            zorder=11
                        )
    
    def _draw_layer_boxes(self, ax: plt.Axes, network: NeuralNetwork) -> None:
        """Draw rounded boxes around layers that have box_around_layer=True in their LayerStyle."""
        for layer_id, positions in self.neuron_positions.items():
            layer = network.get_layer(layer_id)
            
            # Get layer-specific style
            layer_style = self._get_layer_style(layer_id, layer.name)
            
            # Check if this layer should have a box
            if not layer_style.box_around_layer:
                continue
            
            # Calculate bounding box for all neurons in this layer
            if not positions:
                continue
            
            xs = [x for x, y in positions]
            ys = [y for x, y in positions]
            
            min_x = min(xs) - self.config.neuron_radius
            max_x = max(xs) + self.config.neuron_radius
            min_y = min(ys) - self.config.neuron_radius
            max_y = max(ys) + self.config.neuron_radius
            
            # Add padding
            padding = layer_style.box_padding
            min_x -= padding
            max_x += padding
            min_y -= padding
            max_y += padding
            
            # Calculate box dimensions
            width = max_x - min_x
            height = max_y - min_y
            
            # Create rounded rectangle
            from matplotlib.patches import FancyBboxPatch
            
            box = FancyBboxPatch(
                (min_x, min_y),
                width,
                height,
                boxstyle=f"round,pad=0,rounding_size={layer_style.box_corner_radius}",
                facecolor=layer_style.box_fill_color if layer_style.box_fill_color else 'none',
                edgecolor=layer_style.box_edge_color,
                linewidth=layer_style.box_edge_width,
                zorder=5,  # Behind neurons (zorder=10) but in front of connections (zorder=1)
                alpha=0.8 if layer_style.box_fill_color else 1.0
            )
            ax.add_patch(box)
    
    def _draw_linear_connections(self, ax: plt.Axes, network: NeuralNetwork) -> None:
        """Draw connections between layers in a linear network."""
        layer_order = network._layer_order
        
        for i in range(len(layer_order) - 1):
            current_layer_id = layer_order[i]
            next_layer_id = layer_order[i + 1]
            
            current_layer = network.get_layer(current_layer_id)
            current_positions = self.neuron_positions[current_layer_id]
            next_positions = self.neuron_positions[next_layer_id]
            
            # Get collapse info
            current_collapsed = self.collapsed_layers.get(current_layer_id, False)
            next_collapsed = self.collapsed_layers.get(next_layer_id, False)
            current_dots_pos = self.collapsed_info.get(current_layer_id, {}).get('dots_position', -1) if current_collapsed else -1
            next_dots_pos = self.collapsed_info.get(next_layer_id, {}).get('dots_position', -1) if next_collapsed else -1
            
            # Get layer-specific style for connection properties
            layer_style = self._get_layer_style(current_layer_id, current_layer.name)
            
            connection_color = layer_style.connection_color or self.config.connection_color
            connection_linewidth = layer_style.connection_linewidth if layer_style.connection_linewidth is not None else self.config.connection_linewidth
            connection_alpha = layer_style.connection_alpha if layer_style.connection_alpha is not None else self.config.connection_alpha
            
            # Create connections, skipping dots positions
            lines = []
            for idx1, (x1, y1) in enumerate(current_positions):
                # Skip if this is the dots position in source layer
                if current_collapsed and idx1 == current_dots_pos:
                    continue
                    
                for idx2, (x2, y2) in enumerate(next_positions):
                    # Skip if this is the dots position in target layer
                    if next_collapsed and idx2 == next_dots_pos:
                        continue
                    lines.append([(x1, y1), (x2, y2)])
            
            # Draw all connections at once using LineCollection
            if lines:
                lc = LineCollection(
                    lines,
                    colors=connection_color,
                    linewidths=connection_linewidth,
                    alpha=connection_alpha,
                    zorder=1
                )
                ax.add_collection(lc)
    
    def _draw_branching_connections(self, ax: plt.Axes, network: NeuralNetwork) -> None:
        """Draw connections between parent and child layers in a branching network."""
        # Group connections by parent layer to apply layer-specific styles
        for layer_id in network.layers.keys():
            # Get children of this layer
            children = network.get_children(layer_id)
            
            if not children:
                continue
            
            parent_layer = network.get_layer(layer_id)
            parent_positions = self.neuron_positions[layer_id]
            
            # Get collapse info for parent
            parent_collapsed = self.collapsed_layers.get(layer_id, False)
            parent_dots_pos = self.collapsed_info.get(layer_id, {}).get('dots_position', -1) if parent_collapsed else -1
            
            # Get layer-specific style for connection properties
            layer_style = self._get_layer_style(layer_id, parent_layer.name)
            
            connection_color = layer_style.connection_color or self.config.connection_color
            connection_linewidth = layer_style.connection_linewidth if layer_style.connection_linewidth is not None else self.config.connection_linewidth
            connection_alpha = layer_style.connection_alpha if layer_style.connection_alpha is not None else self.config.connection_alpha
            
            lines = []
            # Draw connections to each child
            for child_id in children:
                child_positions = self.neuron_positions[child_id]
                
                # Get collapse info for child
                child_collapsed = self.collapsed_layers.get(child_id, False)
                child_dots_pos = self.collapsed_info.get(child_id, {}).get('dots_position', -1) if child_collapsed else -1
                
                # Create connections, skipping dots positions
                for idx1, (x1, y1) in enumerate(parent_positions):
                    # Skip if this is the dots position in parent layer
                    if parent_collapsed and idx1 == parent_dots_pos:
                        continue
                        
                    for idx2, (x2, y2) in enumerate(child_positions):
                        # Skip if this is the dots position in child layer
                        if child_collapsed and idx2 == child_dots_pos:
                            continue
                        lines.append([(x1, y1), (x2, y2)])
            
            # Draw all connections for this layer at once
            if lines:
                lc = LineCollection(
                    lines,
                    colors=connection_color,
                    linewidths=connection_linewidth,
                    alpha=connection_alpha,
                    zorder=1
                )
                ax.add_collection(lc)
    
    def _set_axis_limits(self, ax: plt.Axes, network: NeuralNetwork) -> None:
        """
        Calculate and set appropriate axis limits based on neuron positions and layer boxes.
        
        Args:
            ax: Matplotlib axes object
            network: NeuralNetwork object to check for layer boxes
        """
        if not self.neuron_positions:
            return
        
        # Collect all x and y coordinates
        all_x = []
        all_y = []
        
        for positions in self.neuron_positions.values():
            for x, y in positions:
                all_x.append(x)
                all_y.append(y)
        
        if not all_x or not all_y:
            return
        
        # Calculate bounds with padding
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        # Base padding (use neuron radius as padding unit)
        padding = self.config.neuron_radius * 3
        
        # Check if any layer has a box and increase padding if needed
        max_box_padding = 0.0
        for layer_id in self.neuron_positions.keys():
            layer = network.get_layer(layer_id)
            layer_style = self._get_layer_style(layer_id, layer.name)
            
            if layer_style.box_around_layer:
                # Account for box padding, edge width, and corner radius
                box_extra = (layer_style.box_padding + 
                           layer_style.box_edge_width * 0.1 + 
                           layer_style.box_corner_radius)
                max_box_padding = max(max_box_padding, box_extra)
        
        # Add extra padding for boxes
        total_padding = padding + max_box_padding
        
        ax.set_xlim(x_min - total_padding, x_max + total_padding)
        ax.set_ylim(y_min - total_padding, y_max + total_padding)
    
    def _add_layer_labels(self, ax: plt.Axes, network: NeuralNetwork) -> None:
        """Add layer names to the plot."""
        for layer_id, (x, y) in self.layer_positions.items():
            layer = network.get_layer(layer_id)
            
            # Create label text
            if layer.name:
                label = layer.name
            else:
                idx = network._layer_order.index(layer_id)
                label = f"Layer {idx}"
            
            # Add neuron count
            if isinstance(layer, FullyConnectedLayer):
                neuron_info = f"\n({layer.num_neurons} neurons)"
                if layer.num_neurons > self.config.max_neurons_per_layer:
                    neuron_info = f"\n({layer.num_neurons} neurons)\n[showing {self.config.max_neurons_per_layer}]"
            else:
                neuron_info = f"\n({layer.get_output_size()} units)"
            
            label += neuron_info
            
            # Position label below the layer
            label_y = y - (len(self.neuron_positions[layer_id]) * self.config.neuron_spacing / 2) - 1.5
            
            ax.text(
                x, label_y,
                label,
                ha='center', va='top',
                fontsize=self.config.layer_name_fontsize,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5),
                zorder=12
            )
    
    def _add_layer_variable_names(self, ax: plt.Axes, network: NeuralNetwork) -> None:
        """Add variable name labels to specified layers."""
        for layer_id, (x, y) in self.layer_positions.items():
            layer = network.get_layer(layer_id)
            
            # Check if this layer has variable names defined (by ID or name)
            variable_label = None
            if layer_id in self.config.layer_variable_names:
                variable_label = self.config.layer_variable_names[layer_id]
            elif layer.name and layer.name in self.config.layer_variable_names:
                variable_label = self.config.layer_variable_names[layer.name]
            
            if variable_label is None:
                continue
            
            # Calculate position based on configuration
            neurons_in_layer = self.neuron_positions[layer_id]
            layer_height = len(neurons_in_layer) * self.config.neuron_spacing
            
            # Get layer style to check for boxes
            layer_style = None
            if layer_id in self.config.layer_styles:
                layer_style = self.config.layer_styles[layer_id]
            elif layer.name and layer.name in self.config.layer_styles:
                layer_style = self.config.layer_styles[layer.name]
            
            # Calculate extra offset for boxes
            box_offset = 0
            if layer_style and layer_style.box_around_layer:
                box_offset = layer_style.box_padding + layer_style.box_edge_width * 0.5
            
            if self.config.layer_variable_names_position == 'above':
                label_x = x
                label_y = y + layer_height / 2 + 0.8 + box_offset
                ha, va = 'center', 'bottom'
            elif self.config.layer_variable_names_position == 'below':
                label_x = x
                label_y = y - layer_height / 2 - 0.8 - box_offset
                ha, va = 'center', 'top'
            else:  # 'side'
                # Determine if this is an input or output layer
                is_input = len(network.get_parents(layer_id)) == 0
                is_output = len(network.get_children(layer_id)) == 0
                
                if is_input:
                    # Place to the left of input layers
                    label_x = x - 1.5 - box_offset
                    label_y = y
                    ha, va = 'right', 'center'
                elif is_output:
                    # Place to the right of output layers
                    label_x = x + 1.5 + box_offset
                    label_y = y
                    ha, va = 'left', 'center'
                else:
                    # For hidden layers, place above
                    label_x = x
                    label_y = y + layer_height / 2 + 0.8 + box_offset
                    ha, va = 'center', 'bottom'
            
            ax.text(
                label_x, label_y,
                variable_label,
                ha=ha, va=va,
                fontsize=self.config.layer_variable_names_fontsize,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='lightgray', alpha=0.7, edgecolor='black'),
                zorder=13
            )


# Convenience function for quick plotting
def plot_network(
    network: NeuralNetwork,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    config: Optional[PlotConfig] = None,
    dpi: int = 300,
    format: Optional[str] = None
) -> plt.Figure:
    """
    Convenience function to plot a neural network.
    
    Args:
        network: NeuralNetwork object to visualize
        title: Optional title for the plot
        save_path: Optional path to save the figure
        show: Whether to display the plot
        config: Optional PlotConfig for customization
        dpi: DPI (dots per inch) for saving the figure (default: 300)
        format: File format ('png', 'svg', 'pdf', etc.). If None, inferred from save_path
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer
        >>> from src.NN_PLOTTING_UTILITIES import plot_network
        >>> 
        >>> nn = NeuralNetwork("My Network")
        >>> nn.add_layer(FullyConnectedLayer(10, name="Input"))
        >>> nn.add_layer(FullyConnectedLayer(5, activation="relu", name="Hidden"))
        >>> nn.add_layer(FullyConnectedLayer(2, activation="softmax", name="Output"))
        >>> 
        >>> plot_network(nn, title="My First Network", save_path="my_network.png")
        >>> plot_network(nn, save_path="network.svg", dpi=150, format="svg")
    """
    plotter = NetworkPlotter(config)
    return plotter.plot_network(network, title, save_path, show, dpi, format)
