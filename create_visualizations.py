#!/usr/bin/env python3
"""
Create visualizations for scene graph pipeline results.

This script creates both 2D diagrams and basic 3D visualizations to show:
1. Object positions in both coordinate systems
2. Spatial relationships between objects and affordances
3. Scene graph structure
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from pathlib import Path

def load_results(results_file):
    """Load pipeline results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)

def create_coordinate_comparison_plot(results, output_dir):
    """Create a plot comparing ARKit and SceneFun3D coordinate systems."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Extract object data
    objects = results['objects']
    colors = ['red', 'green', 'blue']

    # Plot ARKit coordinates (use meter values for better comparison)
    for i, obj in enumerate(objects):
        # Use meter coordinates for consistent scale
        arkit_center = obj.get('center_arkit_m', obj.get('center_arkit', [0,0,0]))
        ax1.scatter(arkit_center[0], arkit_center[1],
                   c=colors[i], s=200, alpha=0.7, label=obj['class'])
        ax1.annotate(obj['class'], (arkit_center[0], arkit_center[1]),
                    xytext=(5, 5), textcoords='offset points')

    ax1.set_title('ARKit Coordinates (meters)')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot SceneFun3D coordinates
    for i, obj in enumerate(objects):
        scenefun3d_center = obj['center_scenefun3d']
        ax2.scatter(scenefun3d_center[0], scenefun3d_center[1],
                   c=colors[i], s=200, alpha=0.7, label=obj['class'])
        ax2.annotate(obj['class'], (scenefun3d_center[0], scenefun3d_center[1]),
                    xytext=(5, 5), textcoords='offset points')

    ax2.set_title('SceneFun3D Coordinates (meters)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'coordinate_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Coordinate comparison saved to: {output_file}")
    plt.close()

def create_3d_scene_visualization(results, output_dir):
    """Create 3D visualization of the scene in SceneFun3D coordinates."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot objects
    objects = results['objects']
    colors = ['red', 'green', 'blue']

    for i, obj in enumerate(objects):
        center = obj['center_scenefun3d']
        # Use meter sizes for correct scale
        size = obj.get('size_m', obj.get('size', [1, 1, 1]))

        # Plot object center
        ax.scatter(center[0], center[1], center[2],
                  c=colors[i], s=300, alpha=0.8, label=f"{obj['class']}")

        # Plot bounding box (simplified as wireframe cube)
        # Calculate box corners
        half_size = np.array(size) / 2
        corners = []
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                for dz in [-1, 1]:
                    corner = np.array(center) + half_size * np.array([dx, dy, dz])
                    corners.append(corner)

        corners = np.array(corners)

        # Draw wireframe box edges
        # Define which corners connect to form edges
        edges = [
            (0, 1), (2, 3), (4, 5), (6, 7),  # x-parallel edges
            (0, 2), (1, 3), (4, 6), (5, 7),  # y-parallel edges
            (0, 4), (1, 5), (2, 6), (3, 7)   # z-parallel edges
        ]

        for edge in edges:
            points = corners[[edge[0], edge[1]]]
            ax.plot3D(points[:, 0], points[:, 1], points[:, 2],
                     colors[i], alpha=0.3, linewidth=1)

    # Plot affordances
    affordances = results['affordances']
    for i, aff in enumerate(affordances):
        center = aff['center']

        # Plot affordance center with larger, more visible marker
        ax.scatter(center[0], center[1], center[2],
                  c='purple', s=250, alpha=0.9, marker='^', edgecolors='darkmagenta', linewidth=2,
                  label=f"Affordance {i+1}" if i == 0 else "")

        # Add coordinate text with full task description
        task_name = aff.get('task_description', 'Unknown task')
        # Keep full task description, use line breaks for long descriptions
        if len(task_name) > 25:
            words = task_name.split()
            mid = len(words) // 2
            task_name = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])

        # More comprehensive affordance info
        aff_id_short = aff['id'][:8]
        coord_text = f"AFFORDANCE: {task_name}\nID: {aff_id_short}\nCoords: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]\nPoints: {aff['point_count']}"
        ax.text(center[0], center[1], center[2] + 0.03, coord_text,
               fontsize=7, ha='center', va='bottom', weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lavender', alpha=0.9, edgecolor='purple', linewidth=1.5))

        # Draw bounding box if min/max coords are available
        if 'min_coords' in aff and 'max_coords' in aff:
            min_coords = aff['min_coords']
            max_coords = aff['max_coords']

            # Calculate box corners
            corners = []
            for dx in [min_coords[0], max_coords[0]]:
                for dy in [min_coords[1], max_coords[1]]:
                    for dz in [min_coords[2], max_coords[2]]:
                        corners.append([dx, dy, dz])

            corners = np.array(corners)

            # Draw wireframe box edges for affordance
            # Define which corners connect to form edges
            edges = [
                (0, 1), (2, 3), (4, 5), (6, 7),  # x-parallel edges
                (0, 2), (1, 3), (4, 6), (5, 7),  # y-parallel edges
                (0, 4), (1, 5), (2, 6), (3, 7)   # z-parallel edges
            ]

            for edge in edges:
                points = corners[[edge[0], edge[1]]]
                ax.plot3D(points[:, 0], points[:, 1], points[:, 2],
                         'purple', alpha=0.4, linewidth=1, linestyle='--')

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('3D Scene Visualization\n(Objects + Affordances in SceneFun3D Coordinates - meters)')
    ax.legend()

    # Set equal aspect ratio including both objects and affordances
    all_points = []
    for obj in objects:
        all_points.append(obj['center_scenefun3d'])
    for aff in affordances:
        all_points.append(aff['center'])

    all_points = np.array(all_points)

    max_range = np.array([
        np.max(all_points[:, i]) - np.min(all_points[:, i])
        for i in range(3)
    ]).max() / 2.0

    mid_x = np.mean(all_points[:, 0])
    mid_y = np.mean(all_points[:, 1])
    mid_z = np.mean(all_points[:, 2])

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    output_file = output_dir / '3d_scene_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"🎯 3D scene visualization saved to: {output_file}")
    plt.close()

def create_scene_graph_diagram(results, output_dir):
    """Create a hierarchical scene graph diagram."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Define positions for hierarchical layout
    scene_pos = (0.5, 0.9)
    object_positions = [(0.2, 0.6), (0.5, 0.6), (0.8, 0.6)]
    affordance_positions = [(0.1, 0.3), (0.3, 0.3), (0.5, 0.3), (0.7, 0.3), (0.9, 0.3)]

    # Colors
    scene_color = 'lightblue'
    object_colors = ['lightcoral', 'lightgreen', 'lightsteelblue']
    affordance_color = 'plum'

    # Draw scene root
    scene_circle = plt.Circle(scene_pos, 0.08, color=scene_color, alpha=0.8)
    ax.add_patch(scene_circle)
    ax.text(scene_pos[0], scene_pos[1], f"Scene\n{results['visit_id']}",
           ha='center', va='center', fontsize=10, weight='bold')

    # Draw objects
    objects = results['objects']
    for i, (obj, pos) in enumerate(zip(objects, object_positions)):
        circle = plt.Circle(pos, 0.06, color=object_colors[i], alpha=0.8)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], f"{obj['class']}\n{obj['id'][:8]}",
               ha='center', va='center', fontsize=8)

        # Draw connection to scene
        ax.plot([scene_pos[0], pos[0]], [scene_pos[1], pos[1]],
               'k-', alpha=0.5, linewidth=2)

        # Add object info box
        size = obj.get('size_m', obj.get('size', [1, 1, 1]))
        info_text = f"Size: {[f'{s:.2f}m' for s in size]}\nCenter: {[f'{c:.2f}' for c in obj['center_scenefun3d']]}"
        ax.text(pos[0], pos[1] - 0.12, info_text,
               ha='center', va='top', fontsize=6,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Draw affordances
    affordances = results['affordances']
    relationships = results.get('relationships', [])

    for i, (aff, pos) in enumerate(zip(affordances, affordance_positions[:len(affordances)])):
        triangle = plt.scatter(pos[0], pos[1], s=1000, c=affordance_color,
                             marker='^', alpha=0.8, edgecolors='darkmagenta', linewidth=2)

        # Use full task description with proper formatting
        task_name = aff.get('task_description', 'Unknown task')

        # Format task name for better readability in triangle
        if len(task_name) > 20:
            words = task_name.split()
            mid = len(words) // 2
            task_name_display = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
        else:
            task_name_display = task_name

        aff_id_short = aff['id'][:8]
        ax.text(pos[0], pos[1], f"{task_name_display}\nID:{aff_id_short}",
               ha='center', va='center', fontsize=6, weight='bold', color='white')

        # Find relationships for this affordance
        aff_relationships = [r for r in relationships if r['affordance_id'] == aff['id']]

        if aff_relationships:
            # Draw connections to related objects with confidence-based styling
            for rel in aff_relationships:
                # Find object position
                obj_idx = next(i for i, obj in enumerate(objects) if obj['id'] == rel['object_id'])
                obj_pos = object_positions[obj_idx]

                confidence = rel['confidence']

                # Color and style based on confidence
                if confidence > 0.7:
                    color, style, width, alpha = 'green', '-', 2, 0.8
                elif confidence > 0.4:
                    color, style, width, alpha = 'orange', '-', 1.5, 0.6
                else:
                    color, style, width, alpha = 'red', '--', 1, 0.4

                ax.plot([obj_pos[0], pos[0]], [obj_pos[1], pos[1]],
                       color=color, linestyle=style, linewidth=width, alpha=alpha)

                # Add confidence label on the connection
                mid_x, mid_y = (obj_pos[0] + pos[0]) / 2, (obj_pos[1] + pos[1]) / 2
                ax.text(mid_x, mid_y, f'{confidence:.2f}',
                       fontsize=5, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))
        else:
            # Draw dashed connection to scene (floating affordances)
            ax.plot([scene_pos[0], pos[0]], [scene_pos[1], pos[1]],
                   'k--', alpha=0.3, linewidth=1)

        # Add comprehensive affordance info with full semantics
        full_task_name = aff.get('task_description', 'Unknown task')
        info_text = f"TASK: {full_task_name}\nPoints: {aff['point_count']}\nCenter: [{aff['center'][0]:.2f}, {aff['center'][1]:.2f}, {aff['center'][2]:.2f}]\nSize: [{aff['size'][0]:.3f}, {aff['size'][1]:.3f}, {aff['size'][2]:.3f}]"
        ax.text(pos[0], pos[1] - 0.10, info_text,
               ha='center', va='top', fontsize=6, weight='bold',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lavender', alpha=0.9, edgecolor='purple', linewidth=1))

    # Add legend
    legend_elements = [
        mpatches.Circle((0, 0), 1, facecolor=scene_color, label='Scene Root'),
        mpatches.Circle((0, 0), 1, facecolor='lightcoral', label='Objects'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=affordance_color,
                  markersize=10, label='Affordances (Tasks)'),
        plt.Line2D([0], [0], color='green', linewidth=2, label='High Confidence (>0.7)'),
        plt.Line2D([0], [0], color='orange', linewidth=1.5, label='Medium Confidence (>0.4)'),
        plt.Line2D([0], [0], color='red', linewidth=1, linestyle='--', label='Low Confidence (<0.4)'),
        plt.Line2D([0], [0], color='black', linewidth=1, linestyle='--', alpha=0.3, label='Floating (No Parent)')
    ]

    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # Add title and info
    ax.set_title('V2.0 Scene Graph Structure\n(Hierarchical with Floating Affordances)',
                fontsize=14, weight='bold', pad=20)

    # Add key features text
    features_text = """Key V2.0 Features:
• Objects transformed to SceneFun3D coordinates
• Affordances stay in native coordinates
• No false fallback connections
• Floating affordances supported"""

    ax.text(0.02, 0.02, features_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='bottom',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    output_file = output_dir / 'scene_graph_structure.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"🏗️ Scene graph structure saved to: {output_file}")
    plt.close()

def create_transformation_validation_plot(results, output_dir):
    """Create validation plots showing the transformation accuracy."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    objects = results['objects']

    # Plot 1: X-Y projection comparison
    for i, obj in enumerate(objects):
        # Use meter coordinates for both
        arkit_xy = obj.get('center_arkit_m', obj.get('center_arkit', [0, 0]))[:2]
        scenefun3d_xy = obj['center_scenefun3d'][:2]

        ax1.scatter(arkit_xy[0], arkit_xy[1], c='red', s=100, alpha=0.7)
        ax1.annotate(f"{obj['class']} (ARKit)", arkit_xy, xytext=(5, 5),
                    textcoords='offset points', fontsize=8)

        ax2.scatter(scenefun3d_xy[0], scenefun3d_xy[1], c='blue', s=100, alpha=0.7)
        ax2.annotate(f"{obj['class']} (SceneFun3D)", scenefun3d_xy, xytext=(5, 5),
                    textcoords='offset points', fontsize=8)

    ax1.set_title('ARKit Coordinate System (X-Y)')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.grid(True, alpha=0.3)

    ax2.set_title('SceneFun3D Coordinate System (X-Y)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Size preservation check (use meter sizes)
    object_names = [obj['class'] for obj in objects]
    sizes = [obj.get('size_m', obj.get('size', [0.1, 0.1, 0.1])) for obj in objects]

    sizes_array = np.array(sizes)
    x_sizes = sizes_array[:, 0]
    y_sizes = sizes_array[:, 1]
    z_sizes = sizes_array[:, 2]

    x_pos = np.arange(len(object_names))
    width = 0.25

    ax3.bar(x_pos - width, x_sizes, width, label='X dimension', alpha=0.8)
    ax3.bar(x_pos, y_sizes, width, label='Y dimension', alpha=0.8)
    ax3.bar(x_pos + width, z_sizes, width, label='Z dimension', alpha=0.8)

    ax3.set_title('Object Sizes (meters)')
    ax3.set_xlabel('Objects')
    ax3.set_ylabel('Size (m)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(object_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Distance matrix between objects
    n_objects = len(objects)
    distance_matrix = np.zeros((n_objects, n_objects))

    for i in range(n_objects):
        for j in range(n_objects):
            if i != j:
                center_i = np.array(objects[i]['center_scenefun3d'])
                center_j = np.array(objects[j]['center_scenefun3d'])
                distance_matrix[i, j] = np.linalg.norm(center_i - center_j)

    im = ax4.imshow(distance_matrix, cmap='viridis')
    ax4.set_title('Inter-object Distances (SceneFun3D)')
    ax4.set_xlabel('Object Index')
    ax4.set_ylabel('Object Index')
    ax4.set_xticks(range(n_objects))
    ax4.set_yticks(range(n_objects))
    ax4.set_xticklabels([obj['class'] for obj in objects], rotation=45)
    ax4.set_yticklabels([obj['class'] for obj in objects])

    # Add colorbar
    plt.colorbar(im, ax=ax4, label='Distance')

    # Add distance values as text
    for i in range(n_objects):
        for j in range(n_objects):
            if i != j:
                text = ax4.text(j, i, f'{distance_matrix[i, j]:.1f}',
                              ha="center", va="center", color="w", fontsize=8)

    plt.tight_layout()
    output_file = output_dir / 'transformation_validation.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Transformation validation saved to: {output_file}")
    plt.close()

def main():
    """Create all visualizations."""
    print("🎨 Creating Pipeline Visualizations")
    print("=" * 50)

    # Setup
    results_file = "results/pipeline_results.json"
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)

    # Load results
    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        print("   Run pipeline.py first")
        return

    results = load_results(results_file)
    print(f"📊 Loaded results: {len(results['objects'])} objects, {len(results['affordances'])} affordances")

    # Create visualizations
    try:
        create_coordinate_comparison_plot(results, output_dir)
        create_3d_scene_visualization(results, output_dir)
        create_scene_graph_diagram(results, output_dir)
        create_transformation_validation_plot(results, output_dir)

        print(f"\n🎉 All visualizations created successfully in: {output_dir}")
        print("📁 Generated files:")
        for file in output_dir.glob("*.png"):
            print(f"   - {file.name}")

    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()