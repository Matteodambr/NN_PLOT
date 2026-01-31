"""
NN_PLOTTING_UTILITIES - A module for visualizing neural network architectures.

This module provides functions to plot neural network structures with neurons
represented as circles and connections as lines.
Currently supports visualization of feedforward neural networks with fully connected layers.
"""

# Check for matplotlib availability
_MATPLOTLIB_AVAILABLE = True
_MATPLOTLIB_CHECK_DONE = False
_MATPLOTLIB_ERROR_MSG = None

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection
    import matplotlib as mpl
except ImportError as e:
    _MATPLOTLIB_AVAILABLE = False
    _MATPLOTLIB_ERROR_MSG = str(e)
    # Create dummy objects to prevent import errors in other parts
    plt = None
    mpatches = None
    mpl = None

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import numpy as np
from PIL import Image
import io
import requests

# Import from the definition utilities
try:
    from .NN_DEFINITION_UTILITIES import (
        NeuralNetwork,
        FullyConnectedLayer,
        VectorInput,
        ImageInput,
        VectorOutput,
        GenericOutput
    )
except ImportError:
    from NN_DEFINITION_UTILITIES import (
        NeuralNetwork,
        FullyConnectedLayer,
        VectorInput,
        ImageInput,
        VectorOutput,
        GenericOutput
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
        show_type: Override global layer_names_show_type for this layer (None uses global setting)
        show_dim: Override global layer_names_show_dim for this layer (None uses global setting)
        show_activation: Override global layer_names_show_activation for this layer (None uses global setting)
        neuron_text_label_alignment: Override global alignment for this layer's neuron labels ('left', 'center', 'right', or None for global)
        max_neurons_to_plot: Override global max_neurons_per_layer for this layer (None uses global setting)
        collapse_neurons_start: Override global collapse_neurons_start for this layer (None uses global setting)
        collapse_neurons_end: Override global collapse_neurons_end for this layer (None uses global setting)
        layer_name_bold: If True, make the first line of layer name (the custom name) bold (None uses global setting)
        variable_name_color: Color for variable name label (None uses default gray box)
        box_include_neuron_labels: If True, extend the box to include neuron text labels (in the label direction)
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
    show_type: Optional[bool] = None
    show_dim: Optional[bool] = None
    show_activation: Optional[bool] = None
    neuron_text_label_alignment: Optional[str] = None
    max_neurons_to_plot: Optional[int] = None
    collapse_neurons_start: Optional[int] = None
    collapse_neurons_end: Optional[int] = None
    layer_name_bold: Optional[bool] = None
    variable_name_color: Optional[str] = None
    box_include_neuron_labels: bool = False  # If True, extend box to include neuron text labels


@dataclass
class LayerGroup:
    """
    Configuration for grouping multiple layers with a bracket and label.
    
    This allows you to visually group layers (e.g., "Encoder", "Decoder") with a
    bracket displayed below the layers and a descriptive label.
    
    Attributes:
        layer_ids: List of layer IDs or layer names to group together
        label: Text label to display below the bracket
        bracket_style: Style of bracket ('curly', 'square', 'straight')
        bracket_color: Color of the bracket lines
        bracket_linewidth: Width/thickness of the bracket lines
        label_fontsize: Font size for the group label text
        label_color: Color of the group label text
        y_offset: Vertical distance below the layers to position the bracket (in plot units)
        bracket_height: Height of the bracket curves/corners (in plot units)
        additional_spacing: Additional spacing below layer labels (if layer names are shown)
    """
    layer_ids: List[str]
    label: str
    bracket_style: str = 'curly'
    bracket_color: str = 'black'
    bracket_linewidth: float = 2.0
    label_fontsize: int = 12
    label_color: str = 'black'
    y_offset: float = -1.5
    bracket_height: float = 0.3
    additional_spacing: float = 0.8


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
        neuron_text_label_alignment: Text alignment for neuron labels ('left', 'center', or 'right')
        neuron_text_label_background: Whether to show a background box behind neuron labels to prevent
                                      connection lines from overlapping the text. Default is True.
                                      Automatically disabled when labels are inside a layer box 
                                      (box_around_layer=True and box_include_neuron_labels=True).
        neuron_text_label_background_padding: Padding around the text in the background box. Default is 0.08.
        neuron_text_label_background_alpha: Transparency of the label background box (0-1). Default is 0.85.
                                            Lower values make connection lines more visible through the text.
        show_layer_names: Whether to show layer names
        show_title: Whether to show the plot title
        title_fontsize: Font size for plot title
        title_offset: Distance (in points) of the title from the top of the plot. Default is 20.
        layer_name_fontsize: Font size for layer names
        max_neurons_per_layer: Maximum neurons to show per layer (for large layers)
        collapse_neurons_start: Number of neurons to show at start when collapsing
        collapse_neurons_end: Number of neurons to show at end when collapsing
        layer_styles: Dictionary mapping layer IDs or names to LayerStyle objects.
                     Use this to apply layer-specific styling including rounded boxes.
        background_color: Background color for the plot. Default is 'white'.
                         Use 'transparent' for transparent background, or any matplotlib color
                         (hex, rgb, named colors).
        layer_groups: List of LayerGroup objects for grouping layers with brackets and labels at the bottom
        layer_spacing_multiplier: Multiplier for the overall network width. Values > 1.0 increase
                                 spacing between layers proportionally, making the network wider.
                                 Default is 1.0 (no scaling). Example: 1.5 makes network 50% wider.
        branch_spacing: Vertical spacing between parallel layers at the same level (in plot units).
                       Controls how close or far apart layers are when the network branches.
                       Default is 3.0. Lower values bring branched layers closer together.
        layer_variable_names: Dictionary mapping layer IDs or names to variable name labels.
                             Example: {'Input': 'Input Variables: x, y, z', 'Output_Head_1': 'Actions: a1, a2'}
        show_layer_variable_names: Whether to show the variable name labels for layers.
        layer_variable_names_fontsize: Font size for layer variable name labels.
        layer_variable_names_position: Position for variable names ('above', 'below', or 'side').
                                      'above' places labels above the layer, 'below' places below,
                                      'side' places to the left for input layers and right for output layers.
        layer_variable_names_wrap: If True, wrap text to fit within max_width. Default is False.
        layer_variable_names_max_width: Maximum width (in characters) for variable name labels before wrapping.
                                       Only applies when layer_variable_names_wrap=True. Default is 20.
        layer_variable_names_multialignment: Text alignment for multi-line variable names ('left', 'center', 'right').
                                            Default is 'center'. Only affects wrapped text with multiple lines.
        layer_variable_names_offset: Distance (in plot units) between the variable name label and the layer edge.
                                    For 'above'/'below' positions, controls vertical spacing. For 'side' position,
                                    controls horizontal spacing. Default is 0.8 for above/below, 1.5 for side.
        layer_names_custom: Dictionary mapping layer IDs or names to custom label text.
                           Example: {'Input Layer': 'Input', 'Hidden_1': 'Encoder', 'Output Layer': 'Output'}
                           This text appears as the first line. If not specified, no custom text is shown.
        layer_names_show_type: If True, show layer type (e.g., 'FC layer' for FullyConnectedLayer) on the second line.
                              If False (default), layer type is not displayed.
        layer_names_show_dim: If True (default), show dimension information (e.g., 'Dim.: 10') on the third line.
                             If False, dimension info is not displayed.
        layer_names_show_activation: If True, show activation function (e.g., 'Act.: ReLU') on the fourth line.
                                    Only displays if the layer has an activation function. Default is False.
        layer_names_align_bottom: If True, all layer name labels appear at the same height at the bottom of the plot.
                                 If False (default), each label appears below its respective layer.
        layer_names_offset: Distance (in plot units) between the bottom of a layer and its label.
                           Only applies when layer_names_align_bottom=False. Default is 0.8.
        layer_names_bottom_offset: Distance (in plot units) of aligned layer labels from the bottom of the plot.
                                  Only applies when layer_names_align_bottom=True. Default is 2.0.
        layer_names_show_box: If True (default), layer name labels have a rounded box background.
                             If False, labels appear as plain text without a box.
        layer_names_line_styles: List of line styles to draw from labels to layers. Options:
                                'vertical_line': Vertical line from label to layer bottom
                                'horizontal_line': Horizontal line with ticks spanning the layer width
                                'brace': Bracket-style brace pointing down from layer to label
                                'curly_brace': Curly brace pointing down from layer to label
                                Can combine multiple styles, e.g., ['vertical_line', 'curly_brace']
                                Default is empty list (no lines).
        layer_names_line_color: Color of the connector lines/braces. Default is 'black'.
        layer_names_line_width: Width of the connector lines/braces. Default is 1.0.
        layer_names_brace_width_multiplier: Multiplier for the width of horizontal_line and brace styles.
                                            Values > 1.0 make braces wider, < 1.0 make them narrower.
                                            Default is 1.0 (spans the layer width plus small margins).
        layer_names_brace_height: Height of the curly brace (in plot units) for single-layer braces.
                                  Default is 0.15. Increase for taller braces, decrease for shorter ones.
        layer_names_brace_label_offset: Distance (in plot units) between the layer name text and the brace/line.
                                        Default is 0.5. Increase for more spacing, decrease for tighter layout.
        font_family: Font family to use for all text in the plot (including math text).
                    Default is 'Times New Roman'. Other options: 'Arial', 'Helvetica', 'DejaVu Sans', etc.
    """
    figsize: Tuple[float, float] = (12, 8)
    neuron_radius: float = 0.3
    layer_spacing: float = 3.0
    neuron_spacing: float = 1.0
    connection_alpha: float = 0.65
    connection_color: str = 'gray'
    connection_linewidth: float = 2.0
    neuron_color: str = 'lightblue'
    neuron_edge_color: str = 'navy'
    neuron_edge_width: float = 1.5
    show_neuron_labels: bool = False
    neuron_numbering_reversed: bool = False
    neuron_label_fontsize: int = 8
    show_neuron_text_labels: bool = True
    neuron_text_label_fontsize: int = 10
    neuron_text_label_offset: float = 0.8
    neuron_text_label_alignment: str = 'center'
    neuron_text_label_background: bool = True
    neuron_text_label_background_padding: float = 0.08
    neuron_text_label_background_alpha: float = 0.75
    show_layer_names: bool = True
    show_title: bool = True
    title_fontsize: int = 16
    title_offset: float = 10
    layer_name_fontsize: int = 12
    max_neurons_per_layer: int = 20
    collapse_neurons_start: int = 10
    collapse_neurons_end: int = 9
    layer_styles: Dict[str, LayerStyle] = field(default_factory=dict)
    background_color: str = 'white'
    layer_spacing_multiplier: float = 1.0
    branch_spacing: float = 3.0
    layer_variable_names: Dict[str, str] = field(default_factory=dict)
    show_layer_variable_names: bool = True
    layer_variable_names_fontsize: int = 11
    layer_variable_names_position: str = 'side'
    layer_variable_names_wrap: bool = False
    layer_variable_names_max_width: int = 20
    layer_variable_names_multialignment: str = 'center'
    layer_variable_names_offset: float = None  # None means use default based on position
    layer_names_custom: Dict[str, str] = field(default_factory=dict)
    layer_names_show_type: bool = False
    layer_names_show_dim: bool = True
    layer_names_show_activation: bool = False
    layer_names_align_bottom: bool = False
    layer_names_offset: float = 0.8  # Increased from 0.4 to prevent overlap
    layer_names_bottom_offset: float = 2.0
    layer_names_show_box: bool = True
    layer_names_line_styles: List[str] = field(default_factory=list)
    layer_names_line_color: str = 'black'
    layer_groups: List[LayerGroup] = field(default_factory=list)
    layer_names_line_width: float = 1.0
    layer_names_brace_width_multiplier: float = 1.0
    layer_names_brace_height: float = 0.15
    layer_names_brace_label_offset: float = 0.5
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
            # Try to use LaTeX for publication-quality math rendering with proper accents
            # First check if LaTeX is available
            import subprocess
            try:
                subprocess.run(['latex', '--version'], capture_output=True, check=True, timeout=1)
                mpl.rcParams['text.usetex'] = True
                mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'
                mpl.rcParams['font.family'] = 'serif'
                mpl.rcParams['font.serif'] = [self.config.font_family]
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                # LaTeX not available, use fallback
                raise Exception("LaTeX not available")
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
        self.image_input_bounds: Dict[str, Tuple[float, float, float, float]] = {}  # Store (x_min, x_max, y_min, y_max) for ImageInput layers
        self.generic_output_boxes: Dict[str, Tuple[float, float]] = {}  # Store GenericOutput box dimensions (width, height)
    
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
    
    def _get_text_dimensions(self, ax: plt.Axes, text: str, fontsize: float) -> Tuple[float, float]:
        """
        Get the actual rendered width and height of text in data coordinates.
        
        Uses matplotlib's text rendering to accurately measure text including
        LaTeX math, subscripts, superscripts, and variable-width fonts.
        
        Args:
            ax: The matplotlib axes to use for measurement
            text: The text string to measure (can include LaTeX)
            fontsize: Font size in points
            
        Returns:
            Tuple of (width, height) in data coordinates
        """
        import re
        try:
            fig = ax.get_figure()
            
            # Create a temporary text object - must be visible for LaTeX to render
            # Place it at a position within the axes limits
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            temp_text = ax.text(xlim[0], ylim[0], text, fontsize=fontsize)
            
            # Force canvas draw to render LaTeX text
            fig.canvas.draw()
            
            # Get the renderer and measure
            renderer = fig.canvas.get_renderer()
            bbox = temp_text.get_window_extent(renderer=renderer)
            
            # Convert to data coordinates
            inv_transform = ax.transData.inverted()
            bbox_data = inv_transform.transform_bbox(bbox)
            
            # Clean up
            temp_text.remove()
            
            return bbox_data.width, bbox_data.height
        except Exception:
            # Fallback to estimation if rendering fails
            # Remove $ delimiters
            s = str(text).replace('$', '')
            # Remove LaTeX commands
            s = re.sub(r'\\[a-zA-Z]+', '', s)
            # Remove braces
            s = s.replace('{', '').replace('}', '')
            # Subscripts/superscripts are smaller, count as 0.6 of a character
            main_chars = s.replace('_', '').replace('^', '')
            sub_super_count = s.count('_') + s.count('^')
            effective_len = len(main_chars) + sub_super_count * 0.6
            char_width = fontsize * 0.015
            char_height = fontsize * 0.02
            return effective_len * char_width, char_height

    def plot_network(
        self,
        network: NeuralNetwork,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        dpi: int = 300,
        format: Optional[str] = None,
        ax: Optional[plt.Axes] = None
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
            ax: Optional matplotlib Axes object for plotting on an existing subplot
            
        Returns:
            matplotlib Figure object
            
        Raises:
            ValueError: If network is empty, has no input layers, or has unsupported layer types
        """
        if network.num_layers() == 0:
            raise ValueError("Cannot plot an empty network")
        
        # Check that network has at least one input layer (root layer)
        if not network.has_input_layer():
            raise ValueError(
                "Network has no input layers. Every network must have at least one root layer. "
                "Use VectorInput for input layers, or ensure your first layer has no parent connections."
            )
        
        # Check if network is linear or branching
        is_linear = network.is_linear()
        
        if is_linear:
            return self._plot_linear_network(network, title, save_path, show, dpi, format, ax)
        else:
            return self._plot_branching_network(network, title, save_path, show, dpi, format, ax)
    
    def _plot_linear_network(
        self,
        network: NeuralNetwork,
        title: Optional[str],
        save_path: Optional[str],
        show: bool,
        dpi: int = 300,
        format: Optional[str] = None,
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """Plot a linear (sequential) neural network."""
        
        # Create figure or use provided axes
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize)
        else:
            fig = ax.get_figure()
        
        # Set background color
        if self.config.background_color != 'transparent':
            if ax is None or ax.get_figure() is not None:
                fig.patch.set_facecolor(self.config.background_color)
            ax.set_facecolor(self.config.background_color)
        
        # Calculate positions for all neurons
        self._calculate_linear_positions(network)
        
        # Set axis limits early so text measurements work correctly
        # (text dimension calculations need proper data coordinate transforms)
        self._set_axis_limits(ax, network)
        
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
        
        # Draw layer groups (brackets and labels)
        if self.config.layer_groups:
            self._draw_layer_groups(ax, network)
        
        # Set title (if enabled)
        if self.config.show_title:
            plot_title = title or f"Neural Network: {network.name}"
            ax.set_title(plot_title, fontsize=self.config.title_fontsize, pad=self.config.title_offset, fontname=self.config.font_family)
        
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
        format: Optional[str] = None,
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """Plot a branching (non-linear) neural network."""
        
        # Create figure or use provided axes
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize)
        else:
            fig = ax.get_figure()
        
        # Set background color
        if self.config.background_color != 'transparent':
            if ax is None or ax.get_figure() is not None:
                fig.patch.set_facecolor(self.config.background_color)
            ax.set_facecolor(self.config.background_color)
        
        # Calculate positions using a layer-based approach
        self._calculate_branching_positions(network)
        
        # Set axis limits early so text measurements work correctly
        # (text dimension calculations need proper data coordinate transforms)
        self._set_axis_limits(ax, network)
        
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
            ax.set_title(plot_title, fontsize=self.config.title_fontsize, pad=self.config.title_offset, fontname=self.config.font_family)
        
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
            
            # Special handling for ImageInput layers - they only need a single position
            # for connection purposes, regardless of their visual representation
            if isinstance(layer, ImageInput):
                # Single position at center for connections
                positions = [(x_pos, 0.0)]
                self.neuron_positions[layer_id] = positions
                self.layer_positions[layer_id] = (x_pos, 0.0)
                self.collapsed_layers[layer_id] = False
                continue
            
            # Special handling for GenericOutput - just store center position
            if isinstance(layer, GenericOutput):
                # For GenericOutput, we just need the center position
                self.neuron_positions[layer_id] = [(x_pos, 0)]  # Single center point
                self.layer_positions[layer_id] = (x_pos, 0)
                continue
            
            # Get actual number of neurons
            if isinstance(layer, (FullyConnectedLayer, VectorOutput)):
                actual_neurons = layer.num_neurons
            else:
                actual_neurons = layer.get_output_size()
            
            # Get collapse settings (check layer-specific first, then global)
            layer_style = self._get_layer_style(layer_id, layer.name)
            max_n = self.config.max_neurons_per_layer
            show_start = self.config.collapse_neurons_start
            show_end = self.config.collapse_neurons_end
            
            if layer_style:
                if layer_style.max_neurons_to_plot is not None:
                    max_n = layer_style.max_neurons_to_plot
                if layer_style.collapse_neurons_start is not None:
                    show_start = layer_style.collapse_neurons_start
                if layer_style.collapse_neurons_end is not None:
                    show_end = layer_style.collapse_neurons_end
            
            # Check if we need to collapse this layer
            if actual_neurons > max_n:
                self.collapsed_layers[layer_id] = True
                # Use configured collapse distribution
                
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
                
                # Special handling for ImageInput layers - they only need a single position
                if isinstance(layer, ImageInput):
                    layer_display_counts[layer_id] = 1
                    self.collapsed_layers[layer_id] = False
                    continue
                
                # Special handling for GenericOutput
                if isinstance(layer, GenericOutput):
                    layer_display_counts[layer_id] = 1  # Single point for center
                    continue
                
                # Get actual number of neurons
                if isinstance(layer, (FullyConnectedLayer, VectorOutput)):
                    actual_neurons = layer.num_neurons
                else:
                    actual_neurons = layer.get_output_size()
                
                # Get collapse settings (check layer-specific first, then global)
                layer_style = self._get_layer_style(layer_id, layer.name)
                max_n = self.config.max_neurons_per_layer
                show_start = self.config.collapse_neurons_start
                show_end = self.config.collapse_neurons_end
                
                if layer_style:
                    if layer_style.max_neurons_to_plot is not None:
                        max_n = layer_style.max_neurons_to_plot
                    if layer_style.collapse_neurons_start is not None:
                        show_start = layer_style.collapse_neurons_start
                    if layer_style.collapse_neurons_end is not None:
                        show_end = layer_style.collapse_neurons_end
                
                # Check if we need to collapse this layer
                if actual_neurons > max_n:
                    self.collapsed_layers[layer_id] = True
                    
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
            
            # Calculate heights for all layers in this level
            layer_heights = {}
            
            # Constants for ImageInput sizing
            DEFAULT_IMAGE_SIZE_MULTIPLIER = 15  # Default size multiplier for ImageInput rectangles
            RGB_CHANNEL_OFFSET_RATIO = 0.15  # Offset ratio for separated RGB channels
            
            for layer_id in layer_ids:
                layer = network.get_layer(layer_id)
                
                # Special handling for ImageInput layers - use actual visual height
                if isinstance(layer, ImageInput):
                    # Calculate the visual height of the ImageInput rectangle
                    if layer.custom_size is not None:
                        size_factor = layer.custom_size
                    else:
                        # Default size based on aspect ratio
                        # Validate height is non-zero to prevent division by zero
                        if layer.height <= 0:
                            raise ValueError(f"ImageInput layer {layer_id} has invalid height: {layer.height}")
                        
                        aspect_ratio = layer.width / layer.height
                        if aspect_ratio > 1:
                            size_factor = self.config.neuron_radius * DEFAULT_IMAGE_SIZE_MULTIPLIER
                        else:
                            size_factor = self.config.neuron_radius * DEFAULT_IMAGE_SIZE_MULTIPLIER / aspect_ratio
                    
                    # For RGB separated channels, account for offset
                    if layer.display_mode == "image" and layer.color_mode == "rgb" and layer.separate_channels:
                        offset = size_factor * RGB_CHANNEL_OFFSET_RATIO
                        total_height = size_factor + 2 * offset
                    else:
                        total_height = size_factor
                    
                    layer_heights[layer_id] = total_height
                else:
                    num_neurons_display = layer_display_counts[layer_id]
                    total_height = (num_neurons_display - 1) * self.config.neuron_spacing
                    layer_heights[layer_id] = total_height
            
            vertical_padding = self.config.branch_spacing  # Extra space between layers at the same level
            
            if level_idx == 0:
                # First level: stack all layers centered around y=0
                # The center of the entire group (from top of topmost to bottom of bottommost) should be at y=0
                total_height_all = sum(layer_heights.values()) + (len(layer_ids) - 1) * vertical_padding
                
                # Start from the top of the group
                current_y = total_height_all / 2
                
                # First pass: calculate positions assuming group starts at current_y
                temp_positions = {}
                temp_layer_positions = {}
                
                for layer_id in layer_ids:
                    layer_height = layer_heights[layer_id]
                    num_neurons_display = layer_display_counts[layer_id]
                    
                    # Calculate the center position of this layer
                    # current_y tracks the top edge, so we subtract half the layer height to get center
                    vertical_offset = current_y - layer_height / 2
                    
                    # For neuron positioning, we need to start at the top of the layer's neuron span
                    # The neurons span from (center - height/2) to (center + height/2)
                    y_start = vertical_offset - layer_height / 2
                    positions = []
                    for j in range(num_neurons_display):
                        y_pos = y_start + j * self.config.neuron_spacing
                        positions.append((x_pos, y_pos))
                    
                    temp_positions[layer_id] = positions
                    temp_layer_positions[layer_id] = (x_pos, vertical_offset)
                    
                    # Move down for next layer (full height + padding)
                    current_y -= (layer_height + vertical_padding)
                
                # Second pass: find actual top and bottom bounds of all layers
                # For ImageInput layers, use their visual bounds; for others, use neuron positions
                layer_tops = []
                layer_bottoms = []
                
                for layer_id in layer_ids:
                    layer = network.get_layer(layer_id)
                    
                    if isinstance(layer, ImageInput):
                        # Use the visual height of the rectangle
                        layer_height = layer_heights[layer_id]
                        center_y = temp_layer_positions[layer_id][1]
                        layer_tops.append(center_y + layer_height / 2)
                        layer_bottoms.append(center_y - layer_height / 2)
                    else:
                        # Use neuron positions
                        if temp_positions[layer_id]:
                            y_positions = [pos[1] for pos in temp_positions[layer_id]]
                            layer_tops.append(max(y_positions))
                            layer_bottoms.append(min(y_positions))
                
                # Calculate the center of the entire group
                if layer_tops and layer_bottoms:
                    group_top = max(layer_tops)
                    group_bottom = min(layer_bottoms)
                    actual_group_center = (group_top + group_bottom) / 2
                    
                    # Calculate offset needed to center the group at y=0
                    centering_offset = -actual_group_center
                    
                    # Third pass: apply the centering offset to all positions
                    for layer_id in layer_ids:
                        # Apply offset to neuron positions
                        self.neuron_positions[layer_id] = [
                            (pos[0], pos[1] + centering_offset) 
                            for pos in temp_positions[layer_id]
                        ]
                        
                        # Apply offset to layer center position
                        old_pos = temp_layer_positions[layer_id]
                        self.layer_positions[layer_id] = (old_pos[0], old_pos[1] + centering_offset)
                else:
                    # Fallback: use temp positions as-is
                    self.neuron_positions.update(temp_positions)
                    self.layer_positions.update(temp_layer_positions)
            else:
                # Group layers by their parent set (layers with same parents should be distributed together)
                from collections import defaultdict
                parent_groups = defaultdict(list)
                for layer_id in layer_ids:
                    parents = tuple(sorted(network.get_parents(layer_id)))
                    parent_groups[parents].append(layer_id)
                
                # For each group of layers sharing the same parents, distribute them around the parent center
                for parents, group_layer_ids in parent_groups.items():
                    # Calculate the center of these parents
                    parent_y_positions = []
                    for parent_id in parents:
                        if parent_id in self.layer_positions:
                            parent_y_positions.append(self.layer_positions[parent_id][1])
                    
                    if parent_y_positions:
                        group_center = (max(parent_y_positions) + min(parent_y_positions)) / 2
                    else:
                        group_center = 0.0
                    
                    # Calculate total height for this group
                    group_heights = [layer_heights[lid] for lid in group_layer_ids]
                    total_group_height = sum(group_heights) + (len(group_layer_ids) - 1) * vertical_padding
                    
                    # Distribute layers in this group, centered on group_center
                    current_y = group_center + total_group_height / 2
                    
                    for layer_id in group_layer_ids:
                        layer_height = layer_heights[layer_id]
                        num_neurons_display = layer_display_counts[layer_id]
                        vertical_offset = current_y - layer_height / 2
                        
                        y_start = vertical_offset - layer_height / 2
                        positions = []
                        for j in range(num_neurons_display):
                            y_pos = y_start + j * self.config.neuron_spacing
                            positions.append((x_pos, y_pos))
                        
                        self.neuron_positions[layer_id] = positions
                        self.layer_positions[layer_id] = (x_pos, vertical_offset)
                        current_y -= (layer_height + vertical_padding)
    
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
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load an image from a file path or URL.
        
        Supports any image format that PIL/Pillow can open, including:
        - JPEG (.jpg, .jpeg)
        - PNG (.png) - including with alpha channel (RGBA)
        - GIF (.gif)
        - BMP (.bmp)
        - TIFF (.tif, .tiff)
        - WebP (.webp)
        - And many more formats supported by PIL
        
        Images with alpha channels (RGBA) are automatically converted to RGB
        with a white background. All other color modes are converted to RGB or
        kept as grayscale (L mode).
        
        Args:
            image_path: Path to local file or URL to image
            
        Returns:
            numpy array with shape (H, W, 3) for RGB or (H, W) for grayscale, or None if failed
        """
        try:
            if image_path.startswith(('http://', 'https://')):
                # Load from URL
                response = requests.get(image_path, timeout=10)
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content))
            else:
                # Load from local file
                img = Image.open(image_path)
            
            # Convert to RGB if it has an alpha channel or is in other mode
            if img.mode == 'RGBA':
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                img = background
            elif img.mode != 'RGB' and img.mode != 'L':
                img = img.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(img)
            return img_array
        except Exception as e:
            print(f"Warning: Could not load image from {image_path}: {e}")
            return None
    
    def _apply_image_transforms(self, img_array: np.ndarray, magnification: float, 
                                translation_x: float, translation_y: float,
                                target_aspect: Optional[float] = None) -> np.ndarray:
        """Apply magnification and translation to an image.
        
        Args:
            img_array: Input image as numpy array
            magnification: Magnification factor (>1 zooms in)
            translation_x: Horizontal offset from center (-1 to 1)
            translation_y: Vertical offset from center (-1 to 1)
            target_aspect: Optional target aspect ratio (width/height) to crop to
            
        Returns:
            Transformed image array
        """
        h, w = img_array.shape[:2]
        
        # Calculate crop dimensions based on magnification
        crop_h = int(h / magnification)
        crop_w = int(w / magnification)
        
        # If target aspect ratio is specified, adjust crop dimensions
        if target_aspect is not None:
            current_aspect = crop_w / crop_h
            if current_aspect > target_aspect:
                # Too wide, reduce width
                crop_w = int(crop_h * target_aspect)
            elif current_aspect < target_aspect:
                # Too tall, reduce height
                crop_h = int(crop_w / target_aspect)
        
        # Calculate center position with translation
        center_x = w / 2 + translation_x * (w / 2)
        center_y = h / 2 + translation_y * (h / 2)
        
        # Calculate crop boundaries
        left = int(max(0, center_x - crop_w / 2))
        right = int(min(w, center_x + crop_w / 2))
        top = int(max(0, center_y - crop_h / 2))
        bottom = int(min(h, center_y + crop_h / 2))
        
        # Crop the image
        cropped = img_array[top:bottom, left:right]
        
        return cropped
    
    def _convert_to_bw(self, img_array: np.ndarray) -> np.ndarray:
        """Convert RGB image to black and white.
        
        Args:
            img_array: Input image as numpy array (H, W, 3) or (H, W)
            
        Returns:
            Black and white image array (H, W, 3) with grayscale values in all channels
        """
        if len(img_array.shape) == 2:
            # Already grayscale, convert to RGB format
            return np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] >= 3:
            # Convert RGB to grayscale using standard weights
            gray = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])
            # Convert back to 3-channel for display
            return np.stack([gray] * 3, axis=-1).astype(np.uint8)
        else:
            return img_array
    
    def _draw_image_input_layer(self, ax: plt.Axes, layer: ImageInput, 
                                layer_id: str, positions: List[Tuple[float, float]]) -> None:
        """Draw an ImageInput layer as a rectangle with image or text.
        
        Args:
            ax: Matplotlib axes
            layer: ImageInput layer object
            layer_id: Layer identifier
            positions: List of neuron positions (used to get center position)
        """
        if not positions:
            return
        
        # Get layer style
        layer_style = self._get_layer_style(layer_id, layer.name)
        
        # Calculate center position - use the center of all neuron positions
        center_x = np.mean([p[0] for p in positions])
        center_y = np.mean([p[1] for p in positions])
        
        # Determine rectangle dimensions based on image aspect ratio
        aspect_ratio = layer.width / layer.height
        
        # Use custom_size if provided, otherwise calculate based on aspect ratio
        if layer.custom_size is not None:
            # Use the custom size directly
            if aspect_ratio > 1:
                # Wider than tall
                rect_width = layer.custom_size * aspect_ratio
                rect_height = layer.custom_size
            else:
                # Taller than wide
                rect_width = layer.custom_size
                rect_height = layer.custom_size / aspect_ratio
        else:
            # Base size for the rectangle (will be scaled by aspect ratio)
            base_size = self.config.neuron_spacing * 2.0  # Make it reasonably sized
            
            if aspect_ratio > 1:
                # Wider than tall
                rect_width = base_size * aspect_ratio
                rect_height = base_size
            else:
                # Taller than wide
                rect_width = base_size
                rect_height = base_size / aspect_ratio
        
        # Store bounds for axis limit calculation
        # Account for RGB channel separation offset if applicable
        if layer.display_mode == 'image' and layer.color_mode == 'rgb' and layer.separate_channels:
            # RGB channels are offset by 0.15 * width/height
            offset_x = rect_width * 0.15
            offset_y = rect_height * 0.15
            x_min = center_x - rect_width/2 - offset_x
            x_max = center_x + rect_width/2 + offset_x
            y_min = center_y - rect_height/2 - offset_y
            y_max = center_y + rect_height/2 + offset_y
        else:
            x_min = center_x - rect_width/2
            x_max = center_x + rect_width/2
            y_min = center_y - rect_height/2
            y_max = center_y + rect_height/2
        
        self.image_input_bounds[layer_id] = (x_min, x_max, y_min, y_max)
        
        # Handle different display modes
        if layer.display_mode == 'text':
            # Draw a single rounded rectangle with text
            self._draw_text_mode_rectangle(ax, layer, center_x, center_y, 
                                          rect_width, rect_height, layer_style)
        
        elif layer.display_mode == 'image':
            # Check if we should separate RGB channels
            if layer.color_mode == 'rgb' and layer.separate_channels:
                # Draw 3 overlapped rectangles, one for each RGB channel
                self._draw_rgb_channels_rectangles(ax, layer, center_x, center_y,
                                                  rect_width, rect_height, layer_style)
            else:
                # Draw a single rectangle with the image
                self._draw_single_image_rectangle(ax, layer, center_x, center_y,
                                                 rect_width, rect_height, layer_style)
    
    def _draw_text_mode_rectangle(self, ax: plt.Axes, layer: ImageInput,
                                  center_x: float, center_y: float,
                                  width: float, height: float,
                                  layer_style) -> None:
        """Draw a rounded rectangle with text for ImageInput in text mode."""
        # Get colors
        fill_color = layer_style.neuron_fill_color or 'lightyellow'
        edge_color = layer_style.neuron_edge_color or self.config.neuron_edge_color
        edge_width = layer_style.neuron_edge_width if layer_style.neuron_edge_width is not None else self.config.neuron_edge_width
        
        # Corner radius
        corner_radius = layer.corner_radius if layer.rounded_corners else 0
        
        # Draw rounded rectangle
        rect = mpatches.FancyBboxPatch(
            (center_x - width/2, center_y - height/2),
            width, height,
            boxstyle=f"round,pad=0,rounding_size={corner_radius}",
            facecolor=fill_color,
            edgecolor=edge_color,
            linewidth=edge_width,
            zorder=10
        )
        ax.add_patch(rect)
        
        # Determine text to display
        if layer.custom_text is not None:
            text = layer.custom_text
            fontsize = layer.custom_text_size
        else:
            # Default text showing dimensions
            text = f"{layer.height}×{layer.width}×{layer.channels}"
            fontsize = 10
        
        # Calculate text bounds to ensure it fits
        # We need to iterate to find the right font size
        TEXT_PADDING_FACTOR = 1.4  # 20% margin on each side (1 + 0.2 + 0.2)
        FONT_SCALE_REDUCTION = 0.95  # Apply 5% reduction for comfortable fit
        
        max_iterations = 10
        for iteration in range(max_iterations):
            # Create temporary text to measure size
            temp_text = ax.text(
                center_x, center_y, text,
                ha='center', va='center',
                fontsize=fontsize,
                fontname=self.config.font_family,
                visible=False  # Make invisible for measurement
            )
            
            # Get text bounding box in data coordinates
            renderer = ax.figure.canvas.get_renderer()
            bbox = temp_text.get_window_extent(renderer=renderer)
            bbox_data = bbox.transformed(ax.transData.inverted())
            text_width = bbox_data.width
            text_height = bbox_data.height
            
            # Remove temporary text
            temp_text.remove()
            
            # Calculate required padding (20% margin on each side for safety)
            required_width = text_width * TEXT_PADDING_FACTOR
            required_height = text_height * TEXT_PADDING_FACTOR
            
            # Check if text fits
            if required_width <= width and required_height <= height:
                break
            
            # Scale down the font size
            scale_factor = min(width / required_width, height / required_height) * FONT_SCALE_REDUCTION
            fontsize = fontsize * scale_factor
        
        # Draw the final text with correct size
        ax.text(
            center_x, center_y, text,
            ha='center', va='center',
            fontsize=fontsize,
            fontname=self.config.font_family,
            zorder=11
        )
    
    def _draw_single_image_rectangle(self, ax: plt.Axes, layer: ImageInput,
                                     center_x: float, center_y: float,
                                     width: float, height: float,
                                     layer_style) -> None:
        """Draw a rectangle with an actual image for ImageInput in single_image mode."""
        # Load the image
        img_array = self._load_image(layer.image_path)
        
        if img_array is None:
            # Fallback to text mode if image can't be loaded
            self._draw_text_mode_rectangle(ax, layer, center_x, center_y, 
                                          width, height, layer_style)
            return
        
        # Convert to BW if requested
        if layer.color_mode == 'bw':
            img_array = self._convert_to_bw(img_array)
        
        # Apply transforms (magnification, translation)
        img_array = self._apply_image_transforms(
            img_array, 
            layer.magnification,
            layer.translation_x,
            layer.translation_y,
            target_aspect=layer.width / layer.height
        )
        
        # Display the image
        extent = [
            center_x - width/2, center_x + width/2,
            center_y - height/2, center_y + height/2
        ]
        im = ax.imshow(img_array, extent=extent, aspect='auto', zorder=10)
        
        # Apply rounded clipping if requested
        if layer.rounded_corners:
            corner_radius = layer.corner_radius
            # Create a rounded rectangle patch for clipping
            clip_rect = mpatches.FancyBboxPatch(
                (center_x - width/2, center_y - height/2),
                width, height,
                boxstyle=f"round,pad=0,rounding_size={corner_radius}",
                transform=ax.transData
            )
            im.set_clip_path(clip_rect)
            
            # Draw border
            edge_color = layer_style.neuron_edge_color or self.config.neuron_edge_color
            edge_width = layer_style.neuron_edge_width if layer_style.neuron_edge_width is not None else self.config.neuron_edge_width
            
            rect = mpatches.FancyBboxPatch(
                (center_x - width/2, center_y - height/2),
                width, height,
                boxstyle=f"round,pad=0,rounding_size={corner_radius}",
                facecolor='none',
                edgecolor=edge_color,
                linewidth=edge_width,
                zorder=11
            )
            ax.add_patch(rect)
    
    def _draw_rgb_channels_rectangles(self, ax: plt.Axes, layer: ImageInput,
                                      center_x: float, center_y: float,
                                      width: float, height: float,
                                      layer_style) -> None:
        """Draw 3 overlapped rectangles for RGB channels in rgb_channels mode."""
        # Load the image
        img_array = self._load_image(layer.image_path)
        
        if img_array is None:
            # Fallback to text mode if image can't be loaded
            self._draw_text_mode_rectangle(ax, layer, center_x, center_y, 
                                          width, height, layer_style)
            return
        
        # Convert to BW if requested (though rgb_channels mode usually expects RGB)
        if layer.color_mode == 'bw':
            img_array = self._convert_to_bw(img_array)
        
        # Apply transforms
        img_array = self._apply_image_transforms(
            img_array,
            layer.magnification,
            layer.translation_x,
            layer.translation_y,
            target_aspect=layer.width / layer.height
        )
        
        # Separate into channels
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            r_channel = img_array[:, :, 0]
            g_channel = img_array[:, :, 1]
            b_channel = img_array[:, :, 2]
        else:
            # Grayscale - use same for all channels
            if len(img_array.shape) == 3:
                grayscale = img_array[:, :, 0]
            else:
                grayscale = img_array
            r_channel = g_channel = b_channel = grayscale
        
        # Offset between rectangles for overlap effect
        offset_x = width * 0.15
        offset_y = height * 0.15
        
        # Draw each channel as a separate rectangle, slightly offset
        channels = [
            (r_channel, 'red', -offset_x, offset_y),      # Red - left/top
            (g_channel, 'green', 0, 0),                    # Green - center
            (b_channel, 'blue', offset_x, -offset_y)       # Blue - right/bottom
        ]
        
        edge_color = layer_style.neuron_edge_color or self.config.neuron_edge_color
        edge_width = layer_style.neuron_edge_width if layer_style.neuron_edge_width is not None else self.config.neuron_edge_width
        corner_radius = layer.corner_radius if layer.rounded_corners else 0
        
        for channel_data, color_name, dx, dy in channels:
            # Create RGB image from single channel with color tint
            if color_name == 'red':
                tinted = np.stack([channel_data, channel_data * 0.3, channel_data * 0.3], axis=-1)
            elif color_name == 'green':
                tinted = np.stack([channel_data * 0.3, channel_data, channel_data * 0.3], axis=-1)
            else:  # blue
                tinted = np.stack([channel_data * 0.3, channel_data * 0.3, channel_data], axis=-1)
            
            tinted = tinted.astype(np.uint8)
            
            # Calculate position for this channel
            ch_center_x = center_x + dx
            ch_center_y = center_y + dy
            
            # Display the channel image
            extent = [
                ch_center_x - width/2, ch_center_x + width/2,
                ch_center_y - height/2, ch_center_y + height/2
            ]
            im = ax.imshow(tinted, extent=extent, aspect='auto', zorder=10, alpha=0.7)
            
            # Apply rounded clipping if requested
            if layer.rounded_corners:
                # Create a rounded rectangle patch for clipping
                clip_rect = mpatches.FancyBboxPatch(
                    (ch_center_x - width/2, ch_center_y - height/2),
                    width, height,
                    boxstyle=f"round,pad=0,rounding_size={corner_radius}",
                    transform=ax.transData
                )
                im.set_clip_path(clip_rect)
                
                # Draw border
                rect = mpatches.FancyBboxPatch(
                    (ch_center_x - width/2, ch_center_y - height/2),
                    width, height,
                    boxstyle=f"round,pad=0,rounding_size={corner_radius}",
                    facecolor='none',
                    edgecolor=edge_color,
                    linewidth=edge_width,
                    zorder=11
                )
                ax.add_patch(rect)
    
    def _draw_neurons(self, ax: plt.Axes, network: NeuralNetwork) -> None:
        """Draw neurons as circles, with ellipsis for collapsed layers.
        
        For ImageInput layers, draws rectangles with images or text instead of neurons.
        GenericOutput layers are drawn as rounded boxes.
        """
        for layer_id, positions in self.neuron_positions.items():
            layer = network.get_layer(layer_id)
            
            # Special handling for ImageInput layers
            if isinstance(layer, ImageInput):
                self._draw_image_input_layer(ax, layer, layer_id, positions)
                continue
            
            # Special handling for GenericOutput - draw as a rounded box with text
            if isinstance(layer, GenericOutput):
                self._draw_generic_output(ax, layer, layer_id, positions[0])
                continue
            
            # Get layer-specific style or use defaults
            layer_style = self._get_layer_style(layer_id, layer.name)
            
            # Determine colors and edge properties
            if isinstance(layer, (FullyConnectedLayer, VectorOutput)):
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
            max_label_width = 0
            max_label_height = 0
            if (self.config.show_neuron_text_labels and 
                isinstance(layer, (FullyConnectedLayer, VectorInput, VectorOutput)) and 
                layer.neuron_labels is not None):
                # Get the x-coordinate of the layer (all neurons in same layer have same x)
                if positions:
                    layer_x = positions[0][0]  # All neurons have same x in a layer
                    if layer.label_position == "left":
                        layer_label_x = layer_x - self.config.neuron_text_label_offset
                    else:  # "right"
                        layer_label_x = layer_x + self.config.neuron_text_label_offset
                
                # Pre-compute maximum label dimensions for uniform background boxes
                # Find the longest label by estimating visual length, then measure that one accurately
                import re
                def estimate_visual_length(s):
                    """Estimate visual length for sorting - longer subscripts = wider text"""
                    s = str(s).replace('$', '')
                    # Count digits in subscripts separately (they're smaller but still take space)
                    # Extract subscript content
                    subscript_match = re.findall(r'_\{?(\d+)\}?', s)
                    subscript_digits = sum(len(m) for m in subscript_match)
                    # Remove LaTeX commands but keep their content length estimate
                    s_clean = re.sub(r'\\[a-zA-Z]+', 'X', s)  # Replace command with single char
                    s_clean = s_clean.replace('{', '').replace('}', '').replace('_', '').replace('^', '')
                    # Subscript digits count as ~0.7 of a regular character
                    return len(s_clean) + subscript_digits * 0.7
                
                # Find the label with maximum estimated visual length
                longest_label = max(layer.neuron_labels, key=estimate_visual_length)
                
                # Measure the longest label
                w, h = self._get_text_dimensions(ax, str(longest_label), self.config.neuron_text_label_fontsize)
                max_label_width = w
                max_label_height = h
                
                # Add safety margin for width (subscripts, accents can affect width more)
                # Height needs less margin as it's more consistent across labels
                max_label_width *= 1.20
                max_label_height *= 1.05  # Minimal height margin
            
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
                            # First few neurons (indices 0 to show_start-1)
                            actual_index = i
                        else:
                            # Last few neurons (after the dots)
                            # These represent the last show_end neurons of the layer
                            # i.e., indices (actual_count - show_end) to (actual_count - 1)
                            offset_from_end = len(positions) - 1 - i
                            actual_index = collapse_info['actual_count'] - 1 - offset_from_end
                    else:
                        actual_index = i
                    
                    # Apply numbering direction
                    if self.config.neuron_numbering_reversed:
                        # Reverse: bottom-to-top (higher indices at top)
                        if isinstance(layer, (FullyConnectedLayer, VectorOutput)):
                            total_neurons = layer.num_neurons
                        else:
                            total_neurons = layer.get_output_size()
                        actual_index = total_neurons - 1 - actual_index
                    
                    # Convert to 1-based indexing for display
                    display_index = actual_index + 1
                    
                    ax.text(
                        x, y, str(display_index),
                        ha='center', va='center',
                        fontsize=self.config.neuron_label_fontsize,
                        fontname=self.config.font_family,
                        zorder=11
                    )
                
                # Add custom text labels if requested (skip for dots position)
                if (self.config.show_neuron_text_labels and 
                    isinstance(layer, (FullyConnectedLayer, VectorInput, VectorOutput)) and 
                    layer.neuron_labels is not None and
                    not (is_collapsed and i == dots_position)):
                    
                    # Determine label index based on whether labels array matches visible or total neurons
                    if len(layer.neuron_labels) == layer.num_neurons:
                        # Full array of labels - index by actual neuron position in the layer
                        if is_collapsed:
                            collapse_info = self.collapsed_info[layer_id]
                            if i < dots_position:
                                # First few neurons (top of layer)
                                label_index = i
                            else:
                                # Last few neurons (bottom of layer, after the dots)
                                # Map to the end of the labels array
                                label_index = layer.num_neurons - (len(positions) - i)
                        else:
                            label_index = i
                    else:
                        # Labels for visible neurons only - index by visible position
                        # Skip the dots position when counting
                        if is_collapsed and i > dots_position:
                            label_index = i - 1  # Account for dots position
                        else:
                            label_index = i
                    
                    # Get the text label for this neuron
                    if 0 <= label_index < len(layer.neuron_labels):
                        label_text = layer.neuron_labels[label_index]
                        
                        # Use pre-calculated label position with center alignment
                        # All labels in the same layer align at the same vertical line
                        label_x = layer_label_x
                        
                        # Determine horizontal alignment: check layer-specific first, then global
                        alignment = self.config.neuron_text_label_alignment
                        
                        # Check for layer-specific alignment override
                        layer_style = self._get_layer_style(layer_id, layer.name)
                        if layer_style and layer_style.neuron_text_label_alignment is not None:
                            alignment = layer_style.neuron_text_label_alignment
                        
                        if alignment not in ['left', 'center', 'right']:
                            alignment = 'center'  # Default to center if invalid
                        
                        # Prepare text kwargs
                        text_kwargs = {
                            'ha': alignment,
                            'va': 'center',
                            'fontsize': self.config.neuron_text_label_fontsize,
                            'zorder': 11
                        }
                        
                        # Add background box if enabled to prevent connection lines overlapping text
                        # Skip if the layer has a box that includes neuron labels (they're already inside a colored box)
                        labels_inside_layer_box = (layer_style.box_around_layer and 
                                                   layer_style.box_include_neuron_labels)
                        
                        if self.config.neuron_text_label_background and not labels_inside_layer_box:
                            # Use the plot background color for the label background
                            bg_color = self.config.background_color
                            if bg_color == 'transparent':
                                bg_color = 'white'  # Default to white for transparent backgrounds
                            
                            # Use pre-computed max dimensions for uniform box sizes across the layer
                            # Add padding for readability (less padding for height to keep boxes compact)
                            padding = self.config.neuron_text_label_background_padding
                            box_width = max_label_width + 2 * padding
                            box_height = max_label_height + padding  # Less vertical padding
                            
                            # Calculate box position based on alignment
                            # The box should be centered around where the WIDEST label would be centered
                            # For center alignment: text is centered at label_x, so box should be too
                            # For left alignment: text left edge is at label_x, box left edge should have padding before it
                            # For right alignment: text right edge is at label_x, box right edge should have padding after it
                            if alignment == 'center':
                                box_x = label_x - box_width / 2
                            elif alignment == 'left':
                                box_x = label_x - padding
                            else:  # 'right'
                                box_x = label_x - max_label_width - padding
                            box_y = y - box_height / 2
                            
                            # Draw background rectangle with rounded corners
                            from matplotlib.patches import FancyBboxPatch
                            bg_box = FancyBboxPatch(
                                (box_x, box_y),
                                box_width,
                                box_height,
                                boxstyle=f'round,pad=0,rounding_size={min(0.1, box_height/4)}',
                                facecolor=bg_color,
                                edgecolor='none',
                                alpha=self.config.neuron_text_label_background_alpha,
                                zorder=10.5  # Above connections but below text
                            )
                            ax.add_patch(bg_box)
                        
                        # Draw the text label with LaTeX support
                        ax.text(label_x, y, label_text,
                                ha=alignment, va='center',
                                fontsize=self.config.neuron_text_label_fontsize,
                                fontname=self.config.font_family,
                                zorder=11)
    
    def _get_generic_output_box_dimensions(self, ax: plt.Axes, layer: GenericOutput) -> Tuple[float, float]:
        """
        Calculate the dimensions needed for a GenericOutput box based on text size.
        
        Args:
            ax: The matplotlib axes to use for text measurement
            layer: The GenericOutput layer
            
        Returns:
            Tuple of (width, height) for the box
        """
        # Measure text dimensions to size the box appropriately
        # Note: Text is rendered as bold, so we need to account for that
        fontsize = self.config.neuron_text_label_fontsize
        
        # Create a temporary bold text to measure accurately
        from matplotlib.font_manager import FontProperties
        try:
            # Try to measure with bold font
            import matplotlib.pyplot as plt
            fig = ax.get_figure()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            temp_text = ax.text(xlim[0], ylim[0], layer.text, 
                               fontsize=fontsize, fontweight='bold')
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox = temp_text.get_window_extent(renderer=renderer)
            inv_transform = ax.transData.inverted()
            bbox_data = inv_transform.transform_bbox(bbox)
            text_width, text_height = bbox_data.width, bbox_data.height
            temp_text.remove()
        except Exception:
            # Fallback: measure normal text and add extra for bold
            text_width, text_height = self._get_text_dimensions(ax, layer.text, fontsize)
            # Bold text is typically ~10-15% wider
            text_width *= 1.15
        
        # Add generous padding around the text
        padding = 0.4  # Increased from 0.3
        box_width = max(text_width + 2 * padding, 1.5)  # Minimum width of 1.5
        box_height = max(text_height + 2 * padding, 0.8)  # Minimum height of 0.8
        
        return box_width, box_height
    
    def _draw_generic_output(self, ax: plt.Axes, layer: GenericOutput, layer_id: str, position: Tuple[float, float]) -> None:
        """Draw a GenericOutput layer as a rounded box with text inside."""
        x, y = position
        
        # Get layer-specific style or use defaults
        layer_style = self._get_layer_style(layer_id, layer.name)
        
        # Calculate box dimensions based on text size
        box_width, box_height = self._get_generic_output_box_dimensions(ax, layer)
        
        # Colors
        fill_color = layer_style.neuron_fill_color or 'lightcoral'
        edge_color = layer_style.neuron_edge_color or self.config.neuron_edge_color
        edge_width = layer_style.neuron_edge_width if layer_style.neuron_edge_width is not None else self.config.neuron_edge_width
        
        # Draw the rounded box
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch(
            (x - box_width/2, y - box_height/2),
            box_width,
            box_height,
            boxstyle='round,pad=0.1',
            facecolor=fill_color,
            edgecolor=edge_color,
            linewidth=edge_width,
            zorder=10
        )
        ax.add_patch(box)
        
        # Draw the text inside the box
        ax.text(
            x, y, layer.text,
            ha='center', va='center',
            fontsize=self.config.neuron_text_label_fontsize,
            fontname=self.config.font_family,
            fontweight='bold',
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
            
            # Extend box to include neuron labels if requested
            if layer_style.box_include_neuron_labels and isinstance(layer, (FullyConnectedLayer, VectorInput, VectorOutput)):
                if layer.neuron_labels is not None and self.config.show_neuron_text_labels:
                    # Labels are positioned at neuron_text_label_offset from neuron center
                    # Box already includes neuron_radius + padding from center
                    
                    fontsize_in_points = self.config.neuron_text_label_fontsize
                    
                    # Find the longest label by estimating visual length
                    import re
                    def estimate_visual_length(s):
                        s = str(s).replace('$', '')
                        subscript_match = re.findall(r'_\{?(\d+)\}?', s)
                        subscript_digits = sum(len(m) for m in subscript_match)
                        s_clean = re.sub(r'\\[a-zA-Z]+', 'X', s)
                        s_clean = s_clean.replace('{', '').replace('}', '').replace('_', '').replace('^', '')
                        return len(s_clean) + subscript_digits * 0.7
                    
                    longest_label = max(layer.neuron_labels, key=estimate_visual_length)
                    max_text_width, _ = self._get_text_dimensions(ax, str(longest_label), fontsize_in_points)
                    # Add safety margin
                    max_text_width *= 1.20
                    
                    # Small fixed margin for readability
                    margin = 0.15
                    
                    # The label is positioned at neuron_text_label_offset from neuron center
                    # For 'center' alignment, the label extends half its width on each side
                    # For 'left' or 'right' alignment, it extends fully to one side
                    alignment = self.config.neuron_text_label_alignment
                    if layer_style.neuron_text_label_alignment is not None:
                        alignment = layer_style.neuron_text_label_alignment
                    
                    if alignment == 'center':
                        # Label centered at offset position, extends half width each direction
                        total_label_extent = self.config.neuron_text_label_offset + max_text_width / 2 + margin
                    elif alignment == 'left':
                        # Label left-aligned at offset position, extends to the right
                        total_label_extent = self.config.neuron_text_label_offset + max_text_width + margin
                    else:  # 'right'
                        # Label right-aligned at offset position, text is to the left of position
                        total_label_extent = self.config.neuron_text_label_offset + margin
                    
                    current_box_extent = self.config.neuron_radius + padding
                    extra_extension = max(0, total_label_extent - current_box_extent)
                    
                    # Extend box in the direction of the labels
                    if layer.label_position == "left":
                        min_x -= extra_extension
                    else:  # "right"
                        max_x += extra_extension
            
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
        
        for layer_id, positions in self.neuron_positions.items():
            layer = network.get_layer(layer_id)
            
            # For GenericOutput, account for the box dimensions
            if isinstance(layer, GenericOutput):
                for x, y in positions:
                    # Calculate box dimensions dynamically based on text
                    box_width, box_height = self._get_generic_output_box_dimensions(ax, layer)
                    all_x.extend([x - box_width/2, x + box_width/2])
                    all_y.extend([y - box_height/2, y + box_height/2])
            else:
                for x, y in positions:
                    all_x.append(x)
                    all_y.append(y)
        
        # Add ImageInput bounds
        for layer_id, (x_min, x_max, y_min, y_max) in self.image_input_bounds.items():
            all_x.extend([x_min, x_max])
            all_y.extend([y_min, y_max])
        
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
        
        # Account for layer group brackets if present
        if self.config.layer_groups:
            # Calculate how much space the group brackets need below
            if self.config.show_layer_names:
                # Estimate space needed for layer labels + group brackets + group labels
                estimated_label_lines = 1
                if self.config.layer_names_show_dim:
                    estimated_label_lines += 1
                if self.config.layer_names_show_activation:
                    estimated_label_lines += 1
                
                layer_label_height = estimated_label_lines * 0.4
                max_spacing = max(group.additional_spacing for group in self.config.layer_groups)
                max_bracket_height = max(group.bracket_height for group in self.config.layer_groups)
                group_label_space = 0.8  # Space for group label text
                
                # Total space needed below
                extra_bottom_space = layer_label_height + max_spacing + max_bracket_height + group_label_space + 1.0
            else:
                # Without layer names, use y_offset and bracket height
                max_bracket_height = max(group.bracket_height for group in self.config.layer_groups)
                group_label_space = 0.8
                extra_bottom_space = abs(self.config.layer_groups[0].y_offset) + max_bracket_height + group_label_space + 0.5
            
            # Adjust y_min to include this space
            y_min = y_min - extra_bottom_space
        
        ax.set_xlim(x_min - total_padding, x_max + total_padding)
        ax.set_ylim(y_min - total_padding, y_max + total_padding)
    
    def _add_layer_labels(self, ax: plt.Axes, network: NeuralNetwork) -> None:
        """Add layer names to the plot."""
        # Activation function capitalization mapping
        activation_display = {
            'relu': 'ReLU',
            'leakyrelu': 'LeakyReLU',
            'prelu': 'PReLU',
            'elu': 'ELU',
            'selu': 'SELU',
            'gelu': 'GELU',
            'softmax': 'Softmax',
            'sigmoid': 'Sigmoid',
            'tanh': 'Tanh',
            'softplus': 'Softplus',
            'softsign': 'Softsign',
            'swish': 'Swish',
            'mish': 'Mish',
            'linear': 'Linear',
            'none': 'None'
        }
        
        for layer_id, (x, y) in self.layer_positions.items():
            layer = network.get_layer(layer_id)
            
            # Get layer-specific style if it exists
            layer_style = None
            if layer_id in self.config.layer_styles:
                layer_style = self.config.layer_styles[layer_id]
            elif layer.name and layer.name in self.config.layer_styles:
                layer_style = self.config.layer_styles[layer.name]
            
            # Determine which settings to use (layer-specific overrides global)
            show_type = layer_style.show_type if (layer_style and layer_style.show_type is not None) else self.config.layer_names_show_type
            show_dim = layer_style.show_dim if (layer_style and layer_style.show_dim is not None) else self.config.layer_names_show_dim
            show_activation = layer_style.show_activation if (layer_style and layer_style.show_activation is not None) else self.config.layer_names_show_activation
            
            # Build label in the specified order:
            # Line 1: Custom description (if specified)
            # Line 2: Layer type (if option is active)
            # Line 3: Dimension (if option is active)
            # Line 4: Activation (if option is active)
            
            label_parts = []
            
            # Line 1: Custom description
            if layer_id in self.config.layer_names_custom:
                label_parts.append(self.config.layer_names_custom[layer_id])
            elif layer.name and layer.name in self.config.layer_names_custom:
                label_parts.append(self.config.layer_names_custom[layer.name])
            elif layer.name and not show_type:
                # If no custom name and show_type is False, use layer.name as fallback
                label_parts.append(layer.name)
            
            # Line 2: Layer type (for MLP layers, show "FC layer")
            if show_type:
                if isinstance(layer, FullyConnectedLayer):
                    label_parts.append("FC layer")
                elif isinstance(layer, ImageInput):
                    # For ImageInput, show "Image Input" with color mode indicator
                    color_indicator = f"({layer.color_mode.upper()})" if layer.color_mode else ""
                    label_parts.append(f"Image Input {color_indicator}".strip())
                elif isinstance(layer, VectorOutput):
                    label_parts.append("Output layer")
                else:
                    # For other layer types, use a generic label or class name
                    idx = network._layer_order.index(layer_id)
                    label_parts.append(f"Layer {idx}")
            
            # Line 3: Dimension information
            if show_dim:
                if isinstance(layer, (FullyConnectedLayer, VectorOutput)):
                    dim_text = f"Dim.: {layer.num_neurons}"
                    label_parts.append(dim_text)
                elif isinstance(layer, ImageInput):
                    # For ImageInput, show dimensions as "width x height"
                    dim_text = f"{layer.width} x {layer.height}"
                    label_parts.append(dim_text)
                else:
                    label_parts.append(f"Dim.: {layer.get_output_size()}")
            
            # Line 4: Activation information
            if show_activation:
                if isinstance(layer, (FullyConnectedLayer, VectorOutput)) and layer.activation:
                    # Use capitalized version if available, otherwise use as-is
                    act_name = activation_display.get(layer.activation.lower(), layer.activation)
                    label_parts.append(f"Act.: {act_name}")
            
            # Join all parts with newlines, but only if there are parts to show
            if not label_parts:
                # Fallback: if nothing is enabled, show layer type
                if isinstance(layer, FullyConnectedLayer):
                    label = "FC layer"
                elif isinstance(layer, VectorOutput):
                    label = "Output layer"
                else:
                    idx = network._layer_order.index(layer_id)
                    label = f"Layer {idx}"
            else:
                label = "\n".join(label_parts)
            
            # Check if first line should be bold
            layer_style = self._get_layer_style(layer_id, layer.name)
            use_bold = False
            if layer_style and layer_style.layer_name_bold is not None:
                use_bold = layer_style.layer_name_bold
            
            # Determine font weight
            fontweight = 'bold' if use_bold else 'normal'
            
            # Position label below the layer
            if self.config.layer_names_align_bottom:
                # Find the minimum y position across all layers to align at bottom
                min_y = float('inf')
                for lid, positions in self.neuron_positions.items():
                    if positions:
                        layer_bottom = min(pos[1] for pos in positions)
                        min_y = min(min_y, layer_bottom)
                label_y = min_y - self.config.layer_names_bottom_offset
            else:
                # Position below each individual layer with dynamic offset based on layer size
                base_offset = self.config.layer_names_offset
                
                # For ImageInput layers, calculate offset from rectangle bounds
                if isinstance(layer, ImageInput):
                    # Get the ImageInput bounds to calculate proper offset
                    if layer_id in self.image_input_bounds:
                        bounds = self.image_input_bounds[layer_id]
                        _, _, bottom_y, _ = bounds
                        # Position label below the rectangle with extra spacing
                        label_y = bottom_y - base_offset
                    else:
                        # Fallback to regular calculation with extra offset
                        label_y = y - base_offset * 2
                
                # For GenericOutput layers, calculate offset from box bounds
                elif isinstance(layer, GenericOutput):
                    # Get the GenericOutput bounds to calculate proper offset
                    if layer_id in self.generic_output_bounds:
                        bounds = self.generic_output_bounds[layer_id]
                        _, _, bottom_y, _ = bounds
                        # Position label below the box with extra spacing
                        label_y = bottom_y - base_offset
                    else:
                        # Fallback if no bounds stored
                        label_y = y - base_offset
                
                else:
                    # Regular layers - calculate offset dynamically based on layer height
                    if layer_id in self.neuron_positions and self.neuron_positions[layer_id]:
                        # Find the bottom of the layer
                        layer_positions = self.neuron_positions[layer_id]
                        min_y = min(pos[1] for pos in layer_positions)
                        
                        # Position label below the bottom neuron with spacing
                        # Use neuron radius for additional spacing
                        label_y = min_y - self.config.neuron_radius - base_offset
                    else:
                        # Fallback if no positions found
                        label_y = y - base_offset
            
            # Configure bbox based on show_box setting
            bbox_props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5) if self.config.layer_names_show_box else None
            
            ax.text(
                x, label_y,
                label,
                ha='center', va='top',
                fontsize=self.config.layer_name_fontsize,
                fontname=self.config.font_family,
                fontweight=fontweight,
                bbox=bbox_props,
                zorder=12
            )
            
            # Draw connector lines if enabled
            if self.config.layer_names_line_styles:
                # Get the bottom of the layer
                layer_positions = self.neuron_positions[layer_id]
                if layer_positions:
                    layer_bottom = min(pos[1] for pos in layer_positions) - self.config.neuron_radius
                    layer_left = min(pos[0] for pos in layer_positions)
                    layer_right = max(pos[0] for pos in layer_positions)
                    
                    # Draw vertical line if requested
                    if 'vertical_line' in self.config.layer_names_line_styles:
                        ax.plot(
                            [x, x],
                            [label_y + 0.3, layer_bottom],
                            color=self.config.layer_names_line_color,
                            linewidth=self.config.layer_names_line_width,
                            linestyle='-',
                            zorder=1
                        )
                    
                    # Draw horizontal line with ticks if requested
                    if 'horizontal_line' in self.config.layer_names_line_styles:
                        # Horizontal line at label level spanning layer width, with small vertical ticks
                        # Account for neuron radius to get the actual visual width of the layer
                        margin = 0.15
                        base_width = (layer_right + self.config.neuron_radius + margin) - (layer_left - self.config.neuron_radius - margin)
                        adjusted_width = base_width * self.config.layer_names_brace_width_multiplier
                        center_x = (layer_left + layer_right) / 2
                        brace_left = center_x - adjusted_width / 2
                        brace_right = center_x + adjusted_width / 2
                        brace_y = label_y + self.config.layer_names_brace_label_offset
                        
                        # Horizontal line
                        ax.plot(
                            [brace_left, brace_right],
                            [brace_y, brace_y],
                            color=self.config.layer_names_line_color,
                            linewidth=self.config.layer_names_line_width,
                            linestyle='-',
                            zorder=1
                        )
                        
                        # Left tick
                        ax.plot(
                            [brace_left, brace_left],
                            [brace_y - 0.15, brace_y + 0.15],
                            color=self.config.layer_names_line_color,
                            linewidth=self.config.layer_names_line_width,
                            linestyle='-',
                            zorder=1
                        )
                        
                        # Right tick
                        ax.plot(
                            [brace_right, brace_right],
                            [brace_y - 0.15, brace_y + 0.15],
                            color=self.config.layer_names_line_color,
                            linewidth=self.config.layer_names_line_width,
                            linestyle='-',
                            zorder=1
                        )
                    
                    # Draw curly brace if requested
                    if 'brace' in self.config.layer_names_line_styles:
                        # Use matplotlib's FancyBboxPatch for a proper curly brace
                        margin = 0.15
                        base_width = (layer_right + self.config.neuron_radius + margin) - (layer_left - self.config.neuron_radius - margin)
                        adjusted_width = base_width * self.config.layer_names_brace_width_multiplier
                        center_x = (layer_left + layer_right) / 2
                        brace_left = center_x - adjusted_width / 2
                        brace_right = center_x + adjusted_width / 2
                        brace_y = label_y + self.config.layer_names_brace_label_offset
                        
                        # Create a brace annotation pointing downward
                        ax.annotate('', xy=((brace_left + brace_right) / 2, label_y),
                                    xytext=((brace_left + brace_right) / 2, brace_y),
                                    arrowprops=dict(
                                        arrowstyle='-',
                                        connectionstyle='arc3,rad=0',
                                        linewidth=0
                                    ))
                        
                        # Draw the brace using multiple small arcs for a curly brace effect
                        from matplotlib.patches import FancyBboxPatch
                        width = brace_right - brace_left
                        
                        # Use matplotlib's bracket annotation
                        ax.annotate('', 
                                   xy=(brace_left, brace_y), 
                                   xytext=(brace_right, brace_y),
                                   arrowprops=dict(
                                       arrowstyle='-',
                                       shrinkA=0, shrinkB=0,
                                       connectionstyle='bar,fraction=-0.15',
                                       linewidth=self.config.layer_names_line_width,
                                       color=self.config.layer_names_line_color
                                   ),
                                   zorder=1)
                    
                    # Draw curly brace if requested
                    if 'curly_brace' in self.config.layer_names_line_styles:
                        # Use unified curly brace method
                        margin = 0.15
                        base_width = (layer_right + self.config.neuron_radius + margin) - (layer_left - self.config.neuron_radius - margin)
                        adjusted_width = base_width * self.config.layer_names_brace_width_multiplier
                        center_x = (layer_left + layer_right) / 2
                        brace_left = center_x - adjusted_width / 2
                        brace_right = center_x + adjusted_width / 2
                        brace_y = label_y + self.config.layer_names_brace_label_offset
                        
                        # Use configurable height
                        height = self.config.layer_names_brace_height
                        
                        self._draw_curly_brace(
                            ax, brace_left, brace_right, brace_y,
                            self.config.layer_names_line_color,
                            self.config.layer_names_line_width,
                            height
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
            
            # Wrap text if enabled
            if self.config.layer_variable_names_wrap:
                import textwrap
                wrapped_lines = textwrap.wrap(variable_label, width=self.config.layer_variable_names_max_width)
                variable_label = '\n'.join(wrapped_lines)
            
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
            
            # Use custom offset or defaults based on position
            vertical_offset = self.config.layer_variable_names_offset if self.config.layer_variable_names_offset is not None else 0.8
            horizontal_offset = self.config.layer_variable_names_offset if self.config.layer_variable_names_offset is not None else 1.5
            
            if self.config.layer_variable_names_position == 'above':
                label_x = x
                label_y = y + layer_height / 2 + vertical_offset + box_offset
                ha, va = 'center', 'bottom'
            elif self.config.layer_variable_names_position == 'below':
                label_x = x
                label_y = y - layer_height / 2 - vertical_offset - box_offset
                ha, va = 'center', 'top'
            else:  # 'side'
                # Determine if this is an input or output layer
                is_input = len(network.get_parents(layer_id)) == 0
                is_output = len(network.get_children(layer_id)) == 0
                
                if is_input:
                    # Place to the left of input layers
                    label_x = x - horizontal_offset - box_offset
                    label_y = y
                    ha, va = 'right', 'center'
                elif is_output:
                    # Place to the right of output layers
                    label_x = x + horizontal_offset + box_offset
                    label_y = y
                    ha, va = 'left', 'center'
                else:
                    # For hidden layers, place above
                    label_x = x
                    label_y = y + layer_height / 2 + vertical_offset + box_offset
                    ha, va = 'center', 'bottom'
            
            # Determine bbox color from layer style or use default
            bbox_color = 'lightgray'
            if layer_style and layer_style.variable_name_color is not None:
                bbox_color = layer_style.variable_name_color
            
            ax.text(
                label_x, label_y,
                variable_label,
                ha=ha, va=va,
                fontsize=self.config.layer_variable_names_fontsize,
                fontname=self.config.font_family,
                fontweight='bold',
                multialignment=self.config.layer_variable_names_multialignment,
                bbox=dict(boxstyle='round,pad=0.6', facecolor=bbox_color, alpha=0.7, edgecolor='black'),
                zorder=13
            )
    
    def _draw_layer_groups(self, ax: plt.Axes, network: NeuralNetwork) -> None:
        """
        Draw brackets and labels to group multiple layers together.
        
        This creates visual groupings at the bottom of the plot with brackets
        (curly, square, round, or straight) and descriptive labels.
        All brackets are drawn at the same height.
        """
        if not self.config.layer_groups:
            return
        
        # Step 1: Calculate the common y-position for ALL group brackets
        if self.config.show_layer_names:
            # Position below layer labels
            if self.config.layer_names_align_bottom:
                # All labels are at same height
                min_y = float('inf')
                for lid, positions in self.neuron_positions.items():
                    if positions:
                        layer_bottom = min(pos[1] for pos in positions)
                        min_y = min(min_y, layer_bottom)
                label_y = min_y - self.config.layer_names_bottom_offset
            else:
                # Labels are at different heights, find the lowest across ALL layers
                label_y = float('inf')
                for lid in self.neuron_positions.keys():
                    if lid in self.layer_positions:
                        layer_y = self.layer_positions[lid][1]
                        positions = self.neuron_positions[lid]
                        individual_label_y = layer_y - (len(positions) * self.config.neuron_spacing / 2) - 1.5
                        label_y = min(label_y, individual_label_y)
            
            # Estimate label height based on configuration
            estimated_label_lines = 1  # Start with custom name or type
            if self.config.layer_names_show_dim:
                estimated_label_lines += 1
            if self.config.layer_names_show_activation:
                estimated_label_lines += 1
            
            label_height = estimated_label_lines * 0.4  # Approximate line height
            
            # Use the maximum additional_spacing from all groups
            max_spacing = max(group.additional_spacing for group in self.config.layer_groups)
            common_y_bracket = label_y - label_height - max_spacing
        else:
            # No layer labels shown, position below the lowest neuron across ALL layers
            all_y_positions = []
            for positions in self.neuron_positions.values():
                for _, y in positions:
                    all_y_positions.append(y)
            
            # Use the y_offset from the first group (or could use min/max)
            y_offset = self.config.layer_groups[0].y_offset if self.config.layer_groups else -1.5
            common_y_bracket = min(all_y_positions) + y_offset
        
        # Step 2: Draw each group's bracket at the common y-position
        for group in self.config.layer_groups:
            # Collect layer information for the group
            layer_data = []  # List of (x_position, layer_id)
            
            for layer_id in group.layer_ids:
                found = False
                # Try to find by ID first
                if layer_id in self.neuron_positions:
                    positions = self.neuron_positions[layer_id]
                    if positions:
                        layer_data.append((positions[0][0], layer_id))
                        found = True
                
                # If not found by ID, try to find by layer name
                if not found:
                    for lid, layer in network.layers.items():
                        if layer.name == layer_id and lid in self.neuron_positions:
                            positions = self.neuron_positions[lid]
                            if positions:
                                layer_data.append((positions[0][0], lid))
                                break
            
            if len(layer_data) < 1:
                continue  # Skip if no valid layers found
            
            # Sort by x-position to get correct left-to-right order
            layer_data.sort(key=lambda item: item[0])
            
            # Calculate proper span for each layer (accounting for neuron radius and margins)
            margin = 0.15
            brace_width_mult = self.config.layer_names_brace_width_multiplier
            
            layer_spans = []
            for center_x, lid in layer_data:
                # Get neuron positions for this layer to determine layer width
                if lid in self.neuron_positions:
                    positions = self.neuron_positions[lid]
                    if positions:
                        layer_left = min(pos[0] for pos in positions)
                        layer_right = max(pos[0] for pos in positions)
                        
                        # Calculate span like single-layer brackets do
                        base_width = (layer_right + self.config.neuron_radius + margin) - \
                                   (layer_left - self.config.neuron_radius - margin)
                        adjusted_width = base_width * brace_width_mult
                        span_left = center_x - adjusted_width / 2
                        span_right = center_x + adjusted_width / 2
                        layer_spans.append((span_left, span_right))
            
            if not layer_spans:
                continue  # Skip if no valid spans
            
            # Get the leftmost and rightmost edges of all layer spans
            x_min = min(span[0] for span in layer_spans)
            x_max = max(span[1] for span in layer_spans)
            
            # Draw the bracket at the common y-position
            self._draw_bracket(
                ax,
                x_min, x_max, common_y_bracket,
                style=group.bracket_style,
                color=group.bracket_color,
                linewidth=group.bracket_linewidth,
                height=group.bracket_height
            )
            
            # Add the label below the bracket
            # Position label below bracket, accounting for bracket height and adding spacing
            label_x = (x_min + x_max) / 2
            label_spacing = 0.3  # Additional spacing between bracket and label
            label_y = common_y_bracket - group.bracket_height - label_spacing
            
            ax.text(
                label_x, label_y, group.label,
                ha='center', va='top',
                fontsize=group.label_fontsize,
                fontname=self.config.font_family,
                color=group.label_color,
                fontweight='bold',
                zorder=10
            )
    
    
    def _draw_curly_brace(self, ax: plt.Axes, x_min: float, x_max: float, y: float,
                          color: str, linewidth: float, height: float) -> None:
        """
        Draw a curly brace using matplotlib paths (unified implementation).
        
        Args:
            ax: Matplotlib axes
            x_min: Left x-coordinate
            x_max: Right x-coordinate
            y: Y-coordinate for the bracket baseline (top of bracket)
            color: Color of the bracket
            linewidth: Width of the bracket lines
            height: Height of bracket (extends downward from y)
        """
        import matplotlib.path as mpath
        import matplotlib.patches as mpatches
        
        width = x_max - x_min
        center_x = (x_min + x_max) / 2
        
        # Curl dimensions - scale with both width and height to avoid gaps in short brackets
        curl_width = min(0.1, width * 0.1, height * 0.6)  # Also constrain by height
        
        # Calculate quarter points for the brace
        quarter = width / 4
        
        # Define vertices for a proper curly brace with curved tips
        # Adjusted for higher outer edges and pointier center
        # Each CURVE4 needs: start point (from previous), control1, control2, endpoint
        verts = [
            # === Segment 1: Left curl (from left endpoint to first quarter) ===
            (x_min, y),  # MOVETO: Left endpoint (start)
            (x_min - curl_width * 0.3, y - height * 0.10),  # CURVE4: Control point 1
            (x_min + curl_width * 0.4, y - height * 0.25),  # CURVE4: Control point 2 (reduced from 0.5)
            (x_min + quarter * 0.8, y - height * 0.40),  # CURVE4: End point (first quarter)
            
            # === Segment 2: Left to middle (from first quarter to center tip) ===
            (center_x - quarter * 0.3, y - height * 0.70),  # CURVE4: Control point 1
            (center_x - curl_width * 0.25, y - height * 0.95),  # CURVE4: Control point 2 (reduced from 0.3)
            (center_x, y - height),  # CURVE4: End point (center tip)
            
            # === Segment 3: Middle to right (from center tip to last quarter) ===
            (center_x + curl_width * 0.25, y - height * 0.95),  # CURVE4: Control point 1 (reduced from 0.3)
            (center_x + quarter * 0.3, y - height * 0.70),  # CURVE4: Control point 2
            (x_max - quarter * 0.8, y - height * 0.40),  # CURVE4: End point (last quarter)
            
            # === Segment 4: Right curl (from last quarter to right endpoint) ===
            (x_max - curl_width * 0.4, y - height * 0.25),  # CURVE4: Control point 1 (reduced from 0.5)
            (x_max + curl_width * 0.3, y - height * 0.10),  # CURVE4: Control point 2
            (x_max, y),  # CURVE4: End point (right endpoint)
        ]
        
        codes = [
            mpath.Path.MOVETO,    # Segment 1: Start at left endpoint
            mpath.Path.CURVE4,    # Segment 1: Control point 1
            mpath.Path.CURVE4,    # Segment 1: Control point 2
            mpath.Path.CURVE4,    # Segment 1: End at first quarter
            mpath.Path.CURVE4,    # Segment 2: Control point 1
            mpath.Path.CURVE4,    # Segment 2: Control point 2
            mpath.Path.CURVE4,    # Segment 2: End at center tip
            mpath.Path.CURVE4,    # Segment 3: Control point 1
            mpath.Path.CURVE4,    # Segment 3: Control point 2
            mpath.Path.CURVE4,    # Segment 3: End at last quarter
            mpath.Path.CURVE4,    # Segment 4: Control point 1
            mpath.Path.CURVE4,    # Segment 4: Control point 2
            mpath.Path.CURVE4,    # Segment 4: End at right endpoint
        ]
        
        path = mpath.Path(verts, codes)
        patch = mpatches.PathPatch(
            path,
            facecolor='none',
            edgecolor=color,
            linewidth=linewidth,
            capstyle='round',  # Smooth line endings
            joinstyle='round',  # Smooth connections between segments
            zorder=10
        )
        ax.add_patch(patch)
    
    def _draw_bracket(self, ax: plt.Axes, x_min: float, x_max: float, y: float,
                      style: str, color: str, linewidth: float, height: float) -> None:
        """
        Draw a bracket of specified style.
        
        Args:
            ax: Matplotlib axes
            x_min: Left x-coordinate
            x_max: Right x-coordinate
            y: Y-coordinate for the bracket baseline (top of bracket)
            style: 'curly', 'square', 'round', or 'straight'
            color: Color of the bracket
            linewidth: Width of the bracket lines
            height: Height of bracket curves/corners (extends downward from y)
        """
        import numpy as np
        
        if style == 'curly':
            # Use unified curly brace implementation
            self._draw_curly_brace(ax, x_min, x_max, y, color, linewidth, height)
            
        elif style == 'square':
            # Square bracket hanging down: ⎡___⎤ shape
            ax.plot([x_min, x_min], [y, y - height], color=color, linewidth=linewidth, zorder=10)
            ax.plot([x_min, x_max], [y - height, y - height], color=color, linewidth=linewidth, zorder=10)
            ax.plot([x_max, x_max], [y - height, y], color=color, linewidth=linewidth, zorder=10)
            
        elif style == 'straight':
            # Simple straight line at the baseline
            ax.plot([x_min, x_max], [y, y], color=color, linewidth=linewidth, zorder=10)
        
        else:
            # Default to straight if unknown style
            ax.plot([x_min, x_max], [y, y], color=color, linewidth=linewidth, zorder=10)


def _check_matplotlib_available():
    """
    Check if matplotlib is available and show a helpful error message if not.
    This check is performed only once (on first call to plot_network).
    """
    global _MATPLOTLIB_CHECK_DONE
    
    if _MATPLOTLIB_CHECK_DONE:
        return
    
    _MATPLOTLIB_CHECK_DONE = True
    
    if not _MATPLOTLIB_AVAILABLE:
        error_msg = """
================================================================================
ERROR: matplotlib is not installed
================================================================================

This library requires matplotlib to generate network visualizations.
Please install it using:

    pip install matplotlib

Then try again.
================================================================================
"""
        raise ImportError(error_msg)


# Convenience function for quick plotting
def plot_network(
    network: NeuralNetwork,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    config: Optional[PlotConfig] = None,
    dpi: int = 300,
    format: Optional[str] = None,
    ax: Optional[plt.Axes] = None
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
        ax: Optional matplotlib Axes object for plotting on an existing subplot
            
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
        >>> 
        >>> # Plot on subplots:
        >>> fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        >>> plot_network(nn, title="Network 1", ax=axes[0])
        >>> plot_network(nn, title="Network 2", ax=axes[1])
        >>> plt.show()
    """
    # Check if matplotlib is available (only checked once)
    _check_matplotlib_available()
    
    # Force show=False when ax is provided (user must call plt.show() explicitly)
    if ax is not None:
        show = False
    
    plotter = NetworkPlotter(config)
    return plotter.plot_network(network, title, save_path, show, dpi, format, ax)
