#!/usr/bin/env python3
"""
Scene Graph Visualization Script

Generates clear, hierarchical visualizations of scene graphs with detailed node labeling
and color coding for different node types.

Usage:
    # Single scene graph
    python scene_graph_visualizer.py --input outputs/scene_graphs/task_2_flush_the_toilet.json

    # All scene graphs in batch mode
    python scene_graph_visualizer.py --input-dir outputs/scene_graphs/ --batch

    # Custom output directory
    python scene_graph_visualizer.py --input-dir outputs/scene_graphs/ --output-dir visualizations/
"""

import json
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np


class SceneGraphVisualizer:
    """Visualizes hierarchical scene graphs with detailed node labeling."""

    # Node type color scheme
    NODE_COLORS = {
        'scene_root': '#1f4e79',      # Deep blue
        'spatial_region': '#2e7d32',   # Green
        'object': '#f57500',           # Orange
        'affordance': '#c62828'        # Red
    }

    # Node shapes for different types
    NODE_SHAPES = {
        'scene_root': 's',    # Square
        'spatial_region': 'h', # Hexagon
        'object': 'o',        # Circle
        'affordance': '^'     # Triangle
    }

    # Node sizes
    NODE_SIZES = {
        'scene_root': 3000,
        'spatial_region': 2000,
        'object': 1500,
        'affordance': 1000
    }

    def __init__(self, verbose: bool = False):
        """Initialize visualizer."""
        self.verbose = verbose

    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[INFO] {message}")

    def load_scene_graph(self, json_file: Path) -> Dict[str, Any]:
        """Load scene graph from JSON file.

        Args:
            json_file: Path to scene graph JSON file

        Returns:
            Scene graph data dictionary
        """
        self.log(f"Loading scene graph from {json_file}")

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            raise ValueError(f"Error loading scene graph from {json_file}: {e}")

    def create_networkx_graph(self, scene_data: Dict[str, Any]) -> nx.DiGraph:
        """Create NetworkX directed graph from scene data.

        Args:
            scene_data: Scene graph data dictionary

        Returns:
            NetworkX directed graph
        """
        self.log("Creating NetworkX graph structure")

        G = nx.DiGraph()
        nodes = scene_data['nodes']

        # Add nodes with attributes
        for node_id, node_data in nodes.items():
            G.add_node(node_id, **node_data)

        # Add edges based on parent-child relationships
        for node_id, node_data in nodes.items():
            for child_id in node_data.get('children_ids', []):
                if child_id in nodes:
                    G.add_edge(node_id, child_id)

        return G

    def get_hierarchical_layout(self, G: nx.DiGraph, root_id: str) -> Dict[str, Tuple[float, float]]:
        """Generate hierarchical layout positions for nodes.

        Args:
            G: NetworkX graph
            root_id: Root node ID

        Returns:
            Dictionary mapping node IDs to (x, y) positions
        """
        self.log("Computing hierarchical layout")

        # Get nodes by level
        levels = {}
        queue = [(root_id, 0)]
        visited = set()

        while queue:
            node_id, level = queue.pop(0)
            if node_id in visited:
                continue

            visited.add(node_id)
            if level not in levels:
                levels[level] = []
            levels[level].append(node_id)

            # Add children to queue
            for child in G.successors(node_id):
                if child not in visited:
                    queue.append((child, level + 1))

        # Position nodes
        pos = {}
        level_height = 3.0  # Vertical spacing between levels

        for level, nodes_at_level in levels.items():
            y = -level * level_height  # Top to bottom

            if len(nodes_at_level) == 1:
                # Single node at center
                pos[nodes_at_level[0]] = (0, y)
            else:
                # Multiple nodes spread horizontally
                node_width = 8.0 / max(1, len(nodes_at_level) - 1) if len(nodes_at_level) > 1 else 0
                start_x = -4.0

                for i, node_id in enumerate(nodes_at_level):
                    x = start_x + i * node_width
                    pos[node_id] = (x, y)

        return pos

    def format_node_label(self, node_data: Dict[str, Any]) -> str:
        """Format detailed label for a node.

        Args:
            node_data: Node data dictionary

        Returns:
            Formatted multi-line label string
        """
        node_type = node_data.get('node_type', 'unknown')
        node_id = node_data.get('node_id', 'unknown')[:8]  # Shortened ID

        if node_type == 'scene_root':
            scene_desc = node_data.get('scene_description', 'Scene')
            visit_id = node_data.get('visit_id', 'N/A')
            video_id = node_data.get('video_id', 'N/A')
            return f"{scene_desc}\nVisit: {visit_id}\nVideo: {video_id}"

        elif node_type == 'spatial_region':
            region_name = node_data.get('region_name', 'Region')
            primary_objects = node_data.get('primary_objects', [])
            objects_str = ', '.join(primary_objects[:2])  # Limit to 2 objects
            if len(primary_objects) > 2:
                objects_str += f" (+{len(primary_objects)-2})"
            return f"{region_name}\nObjects: {objects_str}"

        elif node_type == 'object':
            semantic_class = node_data.get('semantic_class', 'Object')
            volume = node_data.get('volume', 0)
            volume_str = f"{volume/1000:.1f}k" if volume > 1000 else f"{volume:.0f}"
            attributes = node_data.get('attributes', {}).get('attributes', {})
            occlusion = attributes.get('occlusion', 'N/A')
            return f"{semantic_class.title()}\nVolume: {volume_str}mm³\nOcclusion: {occlusion}"

        elif node_type == 'affordance':
            affordance_type = node_data.get('affordance_type', 'Affordance')
            motion_type = node_data.get('motion_type', 'N/A')
            confidence = node_data.get('confidence', 0)
            point_count = node_data.get('point_count', 0)
            return f"{affordance_type}\nMotion: {motion_type}\nConf: {confidence:.2f}\nPoints: {point_count}"

        return f"Node {node_id}\nType: {node_type}"

    def visualize_scene_graph(self, scene_data: Dict[str, Any],
                            output_file: Optional[Path] = None,
                            title: Optional[str] = None,
                            figsize: Tuple[int, int] = (16, 12)) -> None:
        """Create visualization of a single scene graph.

        Args:
            scene_data: Scene graph data dictionary
            output_file: Optional output file path
            title: Optional custom title
            figsize: Figure size tuple
        """
        task_desc = scene_data.get('task_description', 'Scene Graph')
        self.log(f"Visualizing scene graph for: {task_desc}")

        # Create graph
        G = self.create_networkx_graph(scene_data)
        root_id = scene_data.get('root_id')

        if not root_id or root_id not in G.nodes:
            raise ValueError("Root node not found in scene graph")

        # Get layout
        pos = self.get_hierarchical_layout(G, root_id)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')

        # Draw nodes by type
        for node_type in self.NODE_COLORS.keys():
            nodes_of_type = [node for node, data in G.nodes(data=True)
                           if data.get('node_type') == node_type]

            if nodes_of_type:
                node_positions = {node: pos[node] for node in nodes_of_type}

                nx.draw_networkx_nodes(
                    G, node_positions,
                    nodelist=nodes_of_type,
                    node_color=self.NODE_COLORS[node_type],
                    node_shape=self.NODE_SHAPES[node_type],
                    node_size=self.NODE_SIZES[node_type],
                    alpha=0.9,
                    ax=ax
                )

        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            edge_color='#666666',
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            width=2,
            alpha=0.7,
            ax=ax
        )

        # Add node labels
        labels = {}
        for node_id, node_data in G.nodes(data=True):
            labels[node_id] = self.format_node_label(node_data)

        # Position labels slightly offset from nodes
        label_pos = {node: (x, y-0.3) for node, (x, y) in pos.items()}

        nx.draw_networkx_labels(
            G, label_pos,
            labels,
            font_size=9,
            font_weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
            ax=ax
        )

        # Add legend
        legend_elements = [
            mpatches.Patch(color=self.NODE_COLORS['scene_root'], label='Scene Root'),
            mpatches.Patch(color=self.NODE_COLORS['spatial_region'], label='Spatial Region'),
            mpatches.Patch(color=self.NODE_COLORS['object'], label='Object'),
            mpatches.Patch(color=self.NODE_COLORS['affordance'], label='Affordance')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        # Add title and task info
        if title:
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
        else:
            plt.title(f"Scene Graph: {task_desc}", fontsize=16, fontweight='bold', pad=20)

        # Add spatial reasoning chain
        reasoning_chain = scene_data.get('spatial_reasoning_chain', [])
        if reasoning_chain:
            reasoning_text = "Spatial Reasoning:\n" + "\n".join([f"{i+1}. {step}"
                                                               for i, step in enumerate(reasoning_chain)])
            plt.figtext(0.02, 0.02, reasoning_text, fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

        # Add target affordances info
        target_affs = scene_data.get('target_affordances', [])
        if target_affs:
            target_text = f"Target Affordances: {len(target_affs)}"
            plt.figtext(0.98, 0.02, target_text, fontsize=9, ha='right',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))

        # Remove axis
        ax.axis('off')

        # Adjust layout
        plt.tight_layout()

        # Save or show
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            self.log(f"Visualization saved to: {output_file}")
        else:
            plt.show()

        plt.close()

    def visualize_batch(self, scene_graphs: List[Dict[str, Any]],
                       output_file: Optional[Path] = None,
                       title: str = "Scene Graph Comparison") -> None:
        """Create batch visualization of multiple scene graphs.

        Args:
            scene_graphs: List of scene graph data dictionaries
            output_file: Optional output file path
            title: Batch visualization title
        """
        self.log(f"Creating batch visualization for {len(scene_graphs)} scene graphs")

        # Calculate grid layout
        n_graphs = len(scene_graphs)
        cols = min(3, n_graphs)  # Max 3 columns
        rows = (n_graphs + cols - 1) // cols  # Ceiling division

        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))

        # Handle single row/column cases
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i, scene_data in enumerate(scene_graphs):
            if i >= len(axes):
                break

            ax = axes[i]

            # Create mini graph
            G = self.create_networkx_graph(scene_data)
            root_id = scene_data.get('root_id')

            if root_id and root_id in G.nodes:
                pos = self.get_hierarchical_layout(G, root_id)

                # Draw nodes (smaller sizes for batch view)
                for node_type in self.NODE_COLORS.keys():
                    nodes_of_type = [node for node, data in G.nodes(data=True)
                                   if data.get('node_type') == node_type]

                    if nodes_of_type:
                        node_positions = {node: pos[node] for node in nodes_of_type}

                        nx.draw_networkx_nodes(
                            G, node_positions,
                            nodelist=nodes_of_type,
                            node_color=self.NODE_COLORS[node_type],
                            node_shape=self.NODE_SHAPES[node_type],
                            node_size=self.NODE_SIZES[node_type] // 4,  # Smaller for batch
                            alpha=0.9,
                            ax=ax
                        )

                # Draw edges
                nx.draw_networkx_edges(
                    G, pos,
                    edge_color='#666666',
                    arrows=True,
                    arrowsize=10,
                    width=1,
                    alpha=0.6,
                    ax=ax
                )

                # Add simple labels (just node types)
                simple_labels = {node: data.get('node_type', '')[:4].upper()
                               for node, data in G.nodes(data=True)}
                nx.draw_networkx_labels(G, pos, simple_labels, font_size=6, ax=ax)

            # Set subplot title
            task_desc = scene_data.get('task_description', f'Graph {i+1}')
            ax.set_title(task_desc, fontsize=10, fontweight='bold')
            ax.axis('off')

        # Hide unused subplots
        for i in range(n_graphs, len(axes)):
            axes[i].axis('off')

        # Add overall title
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Add legend at figure level
        legend_elements = [
            mpatches.Patch(color=self.NODE_COLORS['scene_root'], label='Scene'),
            mpatches.Patch(color=self.NODE_COLORS['spatial_region'], label='Region'),
            mpatches.Patch(color=self.NODE_COLORS['object'], label='Object'),
            mpatches.Patch(color=self.NODE_COLORS['affordance'], label='Affordance')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95),
                  ncol=4, fontsize=10)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Make room for legend

        # Save or show
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            self.log(f"Batch visualization saved to: {output_file}")
        else:
            plt.show()

        plt.close()


def generate_pipeline_visualizations(scene_graphs_dir: str, output_dir: str = None,
                                    verbose: bool = False) -> bool:
    """Generate visualizations for pipeline integration.

    Args:
        scene_graphs_dir: Directory containing scene graph JSON files
        output_dir: Output directory for visualizations (default: scene_graphs_dir/../visualizations)
        verbose: Enable verbose logging

    Returns:
        True if successful, False otherwise
    """
    try:
        scene_graphs_path = Path(scene_graphs_dir)
        if not scene_graphs_path.exists():
            if verbose:
                print(f"Error: Scene graphs directory {scene_graphs_path} does not exist")
            return False

        # Default output directory is sibling to scene_graphs
        if output_dir is None:
            output_dir = scene_graphs_path.parent / "visualizations"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create visualizer
        visualizer = SceneGraphVisualizer(verbose=verbose)

        # Find all JSON files
        json_files = list(scene_graphs_path.glob("*.json"))
        if not json_files:
            if verbose:
                print(f"Warning: No JSON files found in {scene_graphs_path}")
            return True  # Not an error, just no files to process

        # Load all scene graphs
        scene_graphs = []
        for json_file in sorted(json_files):
            try:
                scene_data = visualizer.load_scene_graph(json_file)
                scene_graphs.append(scene_data)
            except Exception as e:
                if verbose:
                    print(f"Warning: Skipping {json_file}: {e}")

        if not scene_graphs:
            if verbose:
                print("Warning: No valid scene graphs loaded")
            return True

        # Generate individual visualizations
        for i, (scene_data, json_file) in enumerate(zip(scene_graphs, json_files)):
            output_name = json_file.stem + "_graph.png"
            output_file = output_dir / output_name
            visualizer.visualize_scene_graph(scene_data, output_file)
            if verbose:
                print(f"Generated visualization {i+1}/{len(scene_graphs)}: {output_file.name}")

        # Generate batch comparison
        if len(scene_graphs) > 1:
            batch_output = output_dir / "scene_graphs_comparison.png"
            visualizer.visualize_batch(scene_graphs, batch_output)
            if verbose:
                print(f"Generated batch comparison: {batch_output.name}")

        return True

    except Exception as e:
        if verbose:
            print(f"Error generating visualizations: {e}")
        return False


def main():
    """Main entry point for command line usage."""
    parser = argparse.ArgumentParser(description="Visualize hierarchical scene graphs")

    parser.add_argument('--input', type=str,
                       help='Single scene graph JSON file to visualize')
    parser.add_argument('--input-dir', type=str,
                       help='Directory containing scene graph JSON files')
    parser.add_argument('--output-dir', type=str, default='outputs/visualizations',
                       help='Output directory for visualizations (default: outputs/visualizations)')
    parser.add_argument('--format', choices=['png', 'svg', 'pdf'], default='png',
                       help='Output format (default: png)')
    parser.add_argument('--batch', action='store_true',
                       help='Create batch comparison visualization')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    # Validate input arguments
    if not args.input and not args.input_dir:
        parser.error("Must specify either --input or --input-dir")

    if args.input and args.input_dir:
        parser.error("Cannot specify both --input and --input-dir")

    # Create visualizer
    visualizer = SceneGraphVisualizer(verbose=args.verbose)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.input:
            # Single file visualization
            input_file = Path(args.input)
            if not input_file.exists():
                print(f"Error: Input file {input_file} does not exist")
                sys.exit(1)

            scene_data = visualizer.load_scene_graph(input_file)

            # Generate output filename
            output_name = input_file.stem + f"_graph.{args.format}"
            output_file = output_dir / output_name

            visualizer.visualize_scene_graph(scene_data, output_file)
            print(f"Visualization saved to: {output_file}")

        elif args.input_dir:
            # Directory visualization
            input_dir = Path(args.input_dir)
            if not input_dir.exists():
                print(f"Error: Input directory {input_dir} does not exist")
                sys.exit(1)

            # Find all JSON files
            json_files = list(input_dir.glob("*.json"))
            if not json_files:
                print(f"Error: No JSON files found in {input_dir}")
                sys.exit(1)

            # Load all scene graphs
            scene_graphs = []
            for json_file in sorted(json_files):
                try:
                    scene_data = visualizer.load_scene_graph(json_file)
                    scene_graphs.append(scene_data)
                except Exception as e:
                    print(f"Warning: Skipping {json_file}: {e}")

            if not scene_graphs:
                print("Error: No valid scene graphs loaded")
                sys.exit(1)

            if args.batch:
                # Create batch visualization
                output_file = output_dir / f"scene_graphs_comparison.{args.format}"
                visualizer.visualize_batch(scene_graphs, output_file)
                print(f"Batch visualization saved to: {output_file}")
            else:
                # Create individual visualizations
                for i, (scene_data, json_file) in enumerate(zip(scene_graphs, json_files)):
                    output_name = json_file.stem + f"_graph.{args.format}"
                    output_file = output_dir / output_name
                    visualizer.visualize_scene_graph(scene_data, output_file)
                    print(f"Visualization {i+1}/{len(scene_graphs)} saved to: {output_file}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()