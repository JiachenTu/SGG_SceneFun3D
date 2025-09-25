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
import sys
import os

# Add SceneFun3D to path for loading point cloud data
scenefun3d_root = Path("/home/jiachen/scratch/SceneFun3D/scenefun3d")
original_cwd = os.getcwd()
os.chdir(str(scenefun3d_root))
sys.path.insert(0, str(scenefun3d_root))

try:
    from utils.data_parser import DataParser
except ImportError:
    print("Warning: Could not import DataParser - point cloud visualization will be disabled")
    DataParser = None
finally:
    os.chdir(original_cwd)

# Add local utils including point cloud labeler
sys.path.append(str(Path(__file__).parent / "utils"))
try:
    from point_cloud_labeler import create_labeled_point_clouds
except ImportError:
    print("Warning: Could not import point_cloud_labeler")
    create_labeled_point_clouds = None

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

def create_point_cloud_visualization(results, output_dir, data_root="/home/jiachen/scratch/SceneFun3D/alignment/data_examples/scenefun3d"):
    """Create 3D point cloud visualization with overlaid object and affordance bounding boxes."""
    if DataParser is None:
        print("⚠️ Skipping point cloud visualization - DataParser not available")
        return

    try:
        # Load point cloud data
        visit_id = results.get('visit_id', '422203')
        parser = DataParser(data_root)
        laser_scan = parser.get_laser_scan(visit_id)

        pts = np.asarray(laser_scan.points)
        print(f"📊 Loaded point cloud: {len(pts):,} points")

        # Sample up to 20k points for plotting
        n_plot = min(20000, len(pts))
        idx = np.random.choice(len(pts), size=n_plot, replace=False)
        spts = pts[idx]

        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Plot point cloud
        c = None
        if laser_scan.has_colors():
            cols = np.asarray(laser_scan.colors)[idx]
            # matplotlib expects colors in range [0,1]
            ax.scatter(spts[:,0], spts[:,1], spts[:,2], s=0.5, c=cols, alpha=0.6)
        else:
            # color by z value
            sc = ax.scatter(spts[:,0], spts[:,1], spts[:,2], s=0.5, c=spts[:,2], cmap='viridis', alpha=0.6)

        # Overlay object bounding boxes
        objects = results['objects']
        object_colors = ['red', 'orange', 'yellow', 'cyan', 'magenta']

        for i, obj in enumerate(objects):
            center = np.array(obj['center_scenefun3d'])
            size = np.array(obj.get('size_m', obj.get('size', [1, 1, 1])))
            color = object_colors[i % len(object_colors)]

            # Draw object bounding box
            half_size = size / 2
            corners = []
            for dx in [-1, 1]:
                for dy in [-1, 1]:
                    for dz in [-1, 1]:
                        corner = center + half_size * np.array([dx, dy, dz])
                        corners.append(corner)

            corners = np.array(corners)

            # Draw wireframe box edges with thicker lines for visibility
            edges = [
                (0, 1), (2, 3), (4, 5), (6, 7),  # x-parallel edges
                (0, 2), (1, 3), (4, 6), (5, 7),  # y-parallel edges
                (0, 4), (1, 5), (2, 6), (3, 7)   # z-parallel edges
            ]

            for edge in edges:
                points = corners[[edge[0], edge[1]]]
                ax.plot3D(points[:, 0], points[:, 1], points[:, 2],
                         color=color, alpha=0.9, linewidth=3)

            # Add object label with background for better visibility
            ax.text(center[0], center[1], center[2] + size[2]/2 + 0.05,
                   f"OBJECT: {obj['class']}\nID: {obj['id'][:8]}",
                   fontsize=10, ha='center', va='bottom', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8, edgecolor='white'))

        # Overlay affordance bounding boxes
        affordances = results['affordances']

        for i, aff in enumerate(affordances):
            if 'min_coords' in aff and 'max_coords' in aff:
                min_coords = np.array(aff['min_coords'])
                max_coords = np.array(aff['max_coords'])
                center = np.array(aff['center'])

                # Calculate box corners
                corners = []
                for dx in [min_coords[0], max_coords[0]]:
                    for dy in [min_coords[1], max_coords[1]]:
                        for dz in [min_coords[2], max_coords[2]]:
                            corners.append([dx, dy, dz])

                corners = np.array(corners)

                # Draw affordance wireframe box edges with dashed lines
                edges = [
                    (0, 1), (2, 3), (4, 5), (6, 7),  # x-parallel edges
                    (0, 2), (1, 3), (4, 6), (5, 7),  # y-parallel edges
                    (0, 4), (1, 5), (2, 6), (3, 7)   # z-parallel edges
                ]

                for edge in edges:
                    points = corners[[edge[0], edge[1]]]
                    ax.plot3D(points[:, 0], points[:, 1], points[:, 2],
                             color='purple', alpha=0.8, linewidth=2, linestyle='--')

                # Add affordance marker and label
                ax.scatter(center[0], center[1], center[2],
                          c='purple', s=200, alpha=0.9, marker='^',
                          edgecolors='white', linewidth=2)

                task_name = aff.get('task_description', 'Unknown task')
                # Truncate long task names for better display
                if len(task_name) > 30:
                    task_name = task_name[:27] + "..."

                ax.text(center[0], center[1], center[2] + (max_coords[2] - min_coords[2])/2 + 0.05,
                       f"AFFORDANCE: {task_name}\nID: {aff['id'][:8]}\nPoints: {aff['point_count']}",
                       fontsize=8, ha='center', va='bottom', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lavender',
                                alpha=0.9, edgecolor='purple', linewidth=1.5))

        # Add legend for better understanding
        legend_elements = [
            plt.Line2D([0], [0], color='red', linewidth=3, label='Object Bounding Boxes'),
            plt.Line2D([0], [0], color='purple', linewidth=2, linestyle='--', label='Affordance Bounding Boxes'),
            plt.scatter([0], [0], c='gray', s=20, alpha=0.6, label='Point Cloud'),
            plt.scatter([0], [0], c='purple', s=100, marker='^', label='Affordance Centers')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

        # Set labels and title
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.set_zlabel('Z (meters)', fontsize=12)
        ax.set_title('Point Cloud with Object & Affordance Bounding Boxes\n(SceneFun3D Coordinates)',
                    fontsize=14, weight='bold', pad=20)

        # Set equal aspect ratio
        all_points = [spts]
        for obj in objects:
            all_points.append([obj['center_scenefun3d']])
        for aff in affordances:
            all_points.append([aff['center']])

        all_points = np.vstack(all_points)

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

        # Save the visualization
        output_file = output_dir / 'point_cloud_with_bboxes.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"🌟 Point cloud visualization with bounding boxes saved to: {output_file}")
        plt.close()

    except Exception as e:
        print(f"❌ Error creating point cloud visualization: {e}")
        import traceback
        traceback.print_exc()

def calculate_best_view_angle(center, size, points=None):
    """Calculate the best viewing angle for an object or affordance."""
    # Default view angles to try (elevation, azimuth)
    angles = [
        (30, 45),   # Standard 3/4 view
        (45, 135),  # Top-left view
        (20, -45),  # Front-right view
        (60, 0),    # Top view
        (0, 90),    # Side view
    ]

    # For objects with clear orientation, choose based on size
    if size is not None:
        size_array = np.array(size)
        # Find the longest dimension for best side view
        max_dim_idx = np.argmax(size_array)

        if max_dim_idx == 0:  # X is longest - view from Y-Z plane
            return (30, 0)
        elif max_dim_idx == 1:  # Y is longest - view from X-Z plane
            return (30, 90)
        else:  # Z is longest - view from below
            return (60, 45)

    # Default good angle
    return (30, 45)

def create_individual_visualizations(results, output_dir, data_root="/home/jiachen/scratch/SceneFun3D/alignment/data_examples/scenefun3d"):
    """Create individual focused visualizations for each object and affordance."""
    if DataParser is None:
        print("⚠️ Skipping individual visualizations - DataParser not available")
        return

    try:
        # Load point cloud data
        visit_id = results.get('visit_id', '422203')
        parser = DataParser(data_root)
        laser_scan = parser.get_laser_scan(visit_id)
        pts = np.asarray(laser_scan.points)

        # Create individual directories
        individual_dir = output_dir / 'individual'
        objects_dir = individual_dir / 'objects'
        affordances_dir = individual_dir / 'affordances'
        objects_dir.mkdir(parents=True, exist_ok=True)
        affordances_dir.mkdir(parents=True, exist_ok=True)

        print(f"📸 Creating individual visualizations...")

        # Individual object visualizations
        objects = results['objects']
        object_colors = ['red', 'orange', 'yellow', 'cyan', 'magenta']

        for i, obj in enumerate(objects):
            center = np.array(obj['center_scenefun3d'])
            size = np.array(obj.get('size_m', obj.get('size', [1, 1, 1])))
            color = object_colors[i % len(object_colors)]

            # Define region of interest around object (with padding)
            padding = np.maximum(size * 2, [0.5, 0.5, 0.5])  # At least 50cm padding
            min_roi = center - padding
            max_roi = center + padding

            # Filter points within ROI
            mask = ((pts >= min_roi) & (pts <= max_roi)).all(axis=1)
            roi_pts = pts[mask]

            if len(roi_pts) < 100:  # Need minimum points for visualization
                # Expand search if too few points
                padding *= 2
                min_roi = center - padding
                max_roi = center + padding
                mask = ((pts >= min_roi) & (pts <= max_roi)).all(axis=1)
                roi_pts = pts[mask]

            # Sample points if too many
            if len(roi_pts) > 15000:
                idx = np.random.choice(len(roi_pts), size=15000, replace=False)
                roi_pts = roi_pts[idx]

            # Create figure
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Plot ROI point cloud
            if laser_scan.has_colors():
                # Get colors for ROI points
                roi_colors = np.asarray(laser_scan.colors)[mask]
                if len(roi_colors) > 15000:
                    roi_colors = roi_colors[idx]
                ax.scatter(roi_pts[:,0], roi_pts[:,1], roi_pts[:,2],
                          s=1.5, c=roi_colors, alpha=0.7)
            else:
                ax.scatter(roi_pts[:,0], roi_pts[:,1], roi_pts[:,2],
                          s=1.5, c=roi_pts[:,2], cmap='viridis', alpha=0.7)

            # Draw object bounding box with thicker lines
            half_size = size / 2
            corners = []
            for dx in [-1, 1]:
                for dy in [-1, 1]:
                    for dz in [-1, 1]:
                        corner = center + half_size * np.array([dx, dy, dz])
                        corners.append(corner)

            corners = np.array(corners)
            edges = [
                (0, 1), (2, 3), (4, 5), (6, 7),  # x-parallel edges
                (0, 2), (1, 3), (4, 6), (5, 7),  # y-parallel edges
                (0, 4), (1, 5), (2, 6), (3, 7)   # z-parallel edges
            ]

            for edge in edges:
                points = corners[[edge[0], edge[1]]]
                ax.plot3D(points[:, 0], points[:, 1], points[:, 2],
                         color=color, alpha=0.9, linewidth=4)

            # Add object center marker
            ax.scatter(center[0], center[1], center[2],
                      c=color, s=300, alpha=1.0, marker='o',
                      edgecolors='white', linewidth=3)

            # Set best viewing angle
            elev, azim = calculate_best_view_angle(center, size)
            ax.view_init(elev=elev, azim=azim)

            # Set labels and title
            ax.set_xlabel('X (meters)', fontsize=12)
            ax.set_ylabel('Y (meters)', fontsize=12)
            ax.set_zlabel('Z (meters)', fontsize=12)
            ax.set_title(f'Object: {obj["class"]}\nID: {obj["id"][:8]}\nSize: [{size[0]:.2f}m × {size[1]:.2f}m × {size[2]:.2f}m]',
                        fontsize=14, weight='bold', pad=20)

            # Set axis limits to focus on object
            margin = np.maximum(size.max() * 0.6, 0.3)
            ax.set_xlim(center[0] - margin, center[0] + margin)
            ax.set_ylim(center[1] - margin, center[1] + margin)
            ax.set_zlim(center[2] - margin, center[2] + margin)

            # Save individual object visualization
            obj_filename = f"object_{i+1}_{obj['class'].lower().replace(' ', '_')}_{obj['id'][:8]}.png"
            output_file = objects_dir / obj_filename
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"   📦 Object {i+1} ({obj['class']}) saved to: {obj_filename}")
            plt.close()

        # Individual affordance visualizations
        affordances = results['affordances']

        for i, aff in enumerate(affordances):
            center = np.array(aff['center'])

            if 'min_coords' in aff and 'max_coords' in aff:
                min_coords = np.array(aff['min_coords'])
                max_coords = np.array(aff['max_coords'])
                size = max_coords - min_coords
            else:
                size = np.array(aff.get('size', [0.1, 0.1, 0.1]))
                min_coords = center - size/2
                max_coords = center + size/2

            # Define ROI around affordance
            padding = np.maximum(size * 1.5, [0.3, 0.3, 0.3])  # At least 30cm padding
            min_roi = center - padding
            max_roi = center + padding

            # Filter points within ROI
            mask = ((pts >= min_roi) & (pts <= max_roi)).all(axis=1)
            roi_pts = pts[mask]

            if len(roi_pts) < 50:  # Expand if too few points
                padding *= 2
                min_roi = center - padding
                max_roi = center + padding
                mask = ((pts >= min_roi) & (pts <= max_roi)).all(axis=1)
                roi_pts = pts[mask]

            # Sample points if too many
            if len(roi_pts) > 10000:
                idx = np.random.choice(len(roi_pts), size=10000, replace=False)
                roi_pts = roi_pts[idx]

            # Create figure
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Plot ROI point cloud
            if len(roi_pts) > 0:
                ax.scatter(roi_pts[:,0], roi_pts[:,1], roi_pts[:,2],
                          s=2, c=roi_pts[:,2], cmap='viridis', alpha=0.8)

            # Draw affordance bounding box
            corners = []
            for dx in [min_coords[0], max_coords[0]]:
                for dy in [min_coords[1], max_coords[1]]:
                    for dz in [min_coords[2], max_coords[2]]:
                        corners.append([dx, dy, dz])

            corners = np.array(corners)
            edges = [
                (0, 1), (2, 3), (4, 5), (6, 7),  # x-parallel edges
                (0, 2), (1, 3), (4, 6), (5, 7),  # y-parallel edges
                (0, 4), (1, 5), (2, 6), (3, 7)   # z-parallel edges
            ]

            for edge in edges:
                points = corners[[edge[0], edge[1]]]
                ax.plot3D(points[:, 0], points[:, 1], points[:, 2],
                         color='purple', alpha=0.9, linewidth=4, linestyle='-')

            # Add affordance center marker
            ax.scatter(center[0], center[1], center[2],
                      c='purple', s=400, alpha=1.0, marker='^',
                      edgecolors='white', linewidth=3)

            # Set best viewing angle
            elev, azim = calculate_best_view_angle(center, size)
            ax.view_init(elev=elev, azim=azim)

            # Set labels and title
            task_name = aff.get('task_description', 'Unknown task')
            ax.set_xlabel('X (meters)', fontsize=12)
            ax.set_ylabel('Y (meters)', fontsize=12)
            ax.set_zlabel('Z (meters)', fontsize=12)
            ax.set_title(f'Affordance: {task_name}\nID: {aff["id"][:8]}\nPoints: {aff["point_count"]} | Size: [{size[0]:.3f}m × {size[1]:.3f}m × {size[2]:.3f}m]',
                        fontsize=14, weight='bold', pad=20)

            # Set axis limits to focus on affordance
            margin = np.maximum(size.max() * 0.8, 0.2)
            ax.set_xlim(center[0] - margin, center[0] + margin)
            ax.set_ylim(center[1] - margin, center[1] + margin)
            ax.set_zlim(center[2] - margin, center[2] + margin)

            # Save individual affordance visualization
            task_safe = task_name.lower().replace(' ', '_').replace('/', '_')[:20]
            aff_filename = f"affordance_{i+1}_{task_safe}_{aff['id'][:8]}.png"
            output_file = affordances_dir / aff_filename
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"   🎯 Affordance {i+1} ({task_name[:20]}...) saved to: {aff_filename}")
            plt.close()

        print(f"✅ Individual visualizations created: {len(objects)} objects, {len(affordances)} affordances")

    except Exception as e:
        print(f"❌ Error creating individual visualizations: {e}")
        import traceback
        traceback.print_exc()

def create_side_by_side_visualizations(results, output_dir, data_root="/home/jiachen/scratch/SceneFun3D/alignment/data_examples/scenefun3d"):
    """Create side-by-side visualizations showing zoomed view and location in whole scene."""
    if DataParser is None:
        print("⚠️ Skipping side-by-side visualizations - DataParser not available")
        return

    try:
        # Load point cloud data
        visit_id = results.get('visit_id', '422203')
        parser = DataParser(data_root)
        laser_scan = parser.get_laser_scan(visit_id)
        pts = np.asarray(laser_scan.points)

        # Sample full scene for context view (use more points for better context)
        n_context = min(30000, len(pts))
        context_idx = np.random.choice(len(pts), size=n_context, replace=False)
        context_pts = pts[context_idx]

        # Create side-by-side directories
        sidebyside_dir = output_dir / 'side_by_side'
        objects_dir = sidebyside_dir / 'objects'
        affordances_dir = sidebyside_dir / 'affordances'
        objects_dir.mkdir(parents=True, exist_ok=True)
        affordances_dir.mkdir(parents=True, exist_ok=True)

        print(f"🔍 Creating side-by-side visualizations...")

        # Side-by-side object visualizations
        objects = results['objects']
        affordances = results['affordances']
        object_colors = ['red', 'orange', 'yellow', 'cyan', 'magenta']

        for i, obj in enumerate(objects):
            center = np.array(obj['center_scenefun3d'])
            size = np.array(obj.get('size_m', obj.get('size', [1, 1, 1])))
            color = object_colors[i % len(object_colors)]

            # Create figure with two subplots side by side
            fig = plt.figure(figsize=(20, 10))

            # === LEFT PANEL: CONTEXT VIEW ===
            ax_context = fig.add_subplot(121, projection='3d')

            # Plot context point cloud
            if laser_scan.has_colors():
                context_colors = np.asarray(laser_scan.colors)[context_idx]
                ax_context.scatter(context_pts[:,0], context_pts[:,1], context_pts[:,2],
                                 s=0.5, c=context_colors, alpha=0.4)
            else:
                ax_context.scatter(context_pts[:,0], context_pts[:,1], context_pts[:,2],
                                 s=0.5, c=context_pts[:,2], cmap='viridis', alpha=0.4)

            # Draw ALL objects in context with faded colors
            for j, other_obj in enumerate(objects):
                other_center = np.array(other_obj['center_scenefun3d'])
                other_size = np.array(other_obj.get('size_m', other_obj.get('size', [1, 1, 1])))
                other_color = object_colors[j % len(object_colors)]

                # Highlight current object, fade others
                alpha = 1.0 if j == i else 0.3
                linewidth = 4 if j == i else 2

                # Draw bounding box
                half_size = other_size / 2
                corners = []
                for dx in [-1, 1]:
                    for dy in [-1, 1]:
                        for dz in [-1, 1]:
                            corner = other_center + half_size * np.array([dx, dy, dz])
                            corners.append(corner)

                corners = np.array(corners)
                edges = [
                    (0, 1), (2, 3), (4, 5), (6, 7),  # x-parallel edges
                    (0, 2), (1, 3), (4, 6), (5, 7),  # y-parallel edges
                    (0, 4), (1, 5), (2, 6), (3, 7)   # z-parallel edges
                ]

                for edge in edges:
                    points = corners[[edge[0], edge[1]]]
                    ax_context.plot3D(points[:, 0], points[:, 1], points[:, 2],
                                    color=other_color, alpha=alpha, linewidth=linewidth)

                # Add center marker (highlight current object)
                marker_size = 400 if j == i else 100
                ax_context.scatter(other_center[0], other_center[1], other_center[2],
                                 c=other_color, s=marker_size, alpha=alpha, marker='o',
                                 edgecolors='white', linewidth=2)

            # Draw ALL affordances in context with faded colors
            for aff in affordances:
                if 'min_coords' in aff and 'max_coords' in aff:
                    min_coords = np.array(aff['min_coords'])
                    max_coords = np.array(aff['max_coords'])
                    aff_center = np.array(aff['center'])

                    # Draw affordance bounding box (faded)
                    corners = []
                    for dx in [min_coords[0], max_coords[0]]:
                        for dy in [min_coords[1], max_coords[1]]:
                            for dz in [min_coords[2], max_coords[2]]:
                                corners.append([dx, dy, dz])

                    corners = np.array(corners)
                    for edge in edges:
                        points = corners[[edge[0], edge[1]]]
                        ax_context.plot3D(points[:, 0], points[:, 1], points[:, 2],
                                        color='purple', alpha=0.2, linewidth=1, linestyle='--')

                    ax_context.scatter(aff_center[0], aff_center[1], aff_center[2],
                                     c='purple', s=50, alpha=0.3, marker='^')

            # Add highlight box around ROI for current object
            roi_padding = np.maximum(size * 2, [0.5, 0.5, 0.5])
            roi_min = center - roi_padding
            roi_max = center + roi_padding

            # Draw ROI box
            roi_corners = []
            for dx in [roi_min[0], roi_max[0]]:
                for dy in [roi_min[1], roi_max[1]]:
                    for dz in [roi_min[2], roi_max[2]]:
                        roi_corners.append([dx, dy, dz])

            roi_corners = np.array(roi_corners)
            for edge in edges:
                points = roi_corners[[edge[0], edge[1]]]
                ax_context.plot3D(points[:, 0], points[:, 1], points[:, 2],
                                color='lime', alpha=0.8, linewidth=3, linestyle=':')

            # Context view settings
            ax_context.set_title(f'CONTEXT: {obj["class"]} Location in Full Scene\n(Green box shows zoomed region)',
                               fontsize=14, weight='bold')
            ax_context.set_xlabel('X (meters)', fontsize=10)
            ax_context.set_ylabel('Y (meters)', fontsize=10)
            ax_context.set_zlabel('Z (meters)', fontsize=10)

            # Set context view to show full scene
            all_points = context_pts
            max_range = np.array([
                np.max(all_points[:, i]) - np.min(all_points[:, i])
                for i in range(3)
            ]).max() / 2.0

            mid_x = np.mean(all_points[:, 0])
            mid_y = np.mean(all_points[:, 1])
            mid_z = np.mean(all_points[:, 2])

            ax_context.set_xlim(mid_x - max_range, mid_x + max_range)
            ax_context.set_ylim(mid_y - max_range, mid_y + max_range)
            ax_context.set_zlim(mid_z - max_range, mid_z + max_range)

            # === RIGHT PANEL: ZOOMED VIEW ===
            ax_zoom = fig.add_subplot(122, projection='3d')

            # Get ROI points for zoom view
            roi_padding = np.maximum(size * 2, [0.5, 0.5, 0.5])
            min_roi = center - roi_padding
            max_roi = center + roi_padding

            mask = ((pts >= min_roi) & (pts <= max_roi)).all(axis=1)
            roi_pts = pts[mask]

            if len(roi_pts) < 100:
                roi_padding *= 2
                min_roi = center - roi_padding
                max_roi = center + roi_padding
                mask = ((pts >= min_roi) & (pts <= max_roi)).all(axis=1)
                roi_pts = pts[mask]

            # Sample ROI points
            if len(roi_pts) > 15000:
                roi_idx = np.random.choice(len(roi_pts), size=15000, replace=False)
                roi_pts = roi_pts[roi_idx]

            # Plot ROI point cloud
            if laser_scan.has_colors():
                roi_colors = np.asarray(laser_scan.colors)[mask]
                if len(roi_colors) > 15000:
                    roi_colors = roi_colors[roi_idx]
                ax_zoom.scatter(roi_pts[:,0], roi_pts[:,1], roi_pts[:,2],
                               s=2, c=roi_colors, alpha=0.8)
            else:
                ax_zoom.scatter(roi_pts[:,0], roi_pts[:,1], roi_pts[:,2],
                               s=2, c=roi_pts[:,2], cmap='viridis', alpha=0.8)

            # Draw current object bounding box (highlighted)
            half_size = size / 2
            corners = []
            for dx in [-1, 1]:
                for dy in [-1, 1]:
                    for dz in [-1, 1]:
                        corner = center + half_size * np.array([dx, dy, dz])
                        corners.append(corner)

            corners = np.array(corners)
            for edge in edges:
                points = corners[[edge[0], edge[1]]]
                ax_zoom.plot3D(points[:, 0], points[:, 1], points[:, 2],
                              color=color, alpha=0.9, linewidth=5)

            # Add object center marker
            ax_zoom.scatter(center[0], center[1], center[2],
                           c=color, s=400, alpha=1.0, marker='o',
                           edgecolors='white', linewidth=3)

            # Draw any affordances that fall within ROI
            for aff in affordances:
                if 'min_coords' in aff and 'max_coords' in aff:
                    aff_center = np.array(aff['center'])
                    # Check if affordance center is within ROI
                    if ((aff_center >= min_roi) & (aff_center <= max_roi)).all():
                        min_coords = np.array(aff['min_coords'])
                        max_coords = np.array(aff['max_coords'])

                        # Draw affordance bounding box
                        corners = []
                        for dx in [min_coords[0], max_coords[0]]:
                            for dy in [min_coords[1], max_coords[1]]:
                                for dz in [min_coords[2], max_coords[2]]:
                                    corners.append([dx, dy, dz])

                        corners = np.array(corners)
                        for edge in edges:
                            points = corners[[edge[0], edge[1]]]
                            ax_zoom.plot3D(points[:, 0], points[:, 1], points[:, 2],
                                         color='purple', alpha=0.7, linewidth=3, linestyle='--')

                        ax_zoom.scatter(aff_center[0], aff_center[1], aff_center[2],
                                       c='purple', s=200, alpha=0.9, marker='^',
                                       edgecolors='white', linewidth=2)

            # Zoom view settings
            elev, azim = calculate_best_view_angle(center, size)
            ax_zoom.view_init(elev=elev, azim=azim)

            ax_zoom.set_title(f'ZOOM: {obj["class"]} Detail View\nSize: [{size[0]:.2f}m × {size[1]:.2f}m × {size[2]:.2f}m]',
                            fontsize=14, weight='bold')
            ax_zoom.set_xlabel('X (meters)', fontsize=10)
            ax_zoom.set_ylabel('Y (meters)', fontsize=10)
            ax_zoom.set_zlabel('Z (meters)', fontsize=10)

            # Set zoom limits
            margin = np.maximum(size.max() * 0.6, 0.3)
            ax_zoom.set_xlim(center[0] - margin, center[0] + margin)
            ax_zoom.set_ylim(center[1] - margin, center[1] + margin)
            ax_zoom.set_zlim(center[2] - margin, center[2] + margin)

            # Add overall title
            fig.suptitle(f'Side-by-Side View: Object {i+1} - {obj["class"]} (ID: {obj["id"][:8]})',
                        fontsize=16, weight='bold', y=0.95)

            plt.tight_layout()

            # Save side-by-side object visualization
            obj_filename = f"sidebyside_object_{i+1}_{obj['class'].lower().replace(' ', '_')}_{obj['id'][:8]}.png"
            output_file = objects_dir / obj_filename
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"   📦🔍 Object {i+1} ({obj['class']}) side-by-side saved to: {obj_filename}")
            plt.close()

        # Side-by-side affordance visualizations
        for i, aff in enumerate(affordances):
            center = np.array(aff['center'])

            if 'min_coords' in aff and 'max_coords' in aff:
                min_coords = np.array(aff['min_coords'])
                max_coords = np.array(aff['max_coords'])
                size = max_coords - min_coords
            else:
                size = np.array(aff.get('size', [0.1, 0.1, 0.1]))
                min_coords = center - size/2
                max_coords = center + size/2

            # Create figure with two subplots
            fig = plt.figure(figsize=(20, 10))

            # === LEFT PANEL: CONTEXT VIEW ===
            ax_context = fig.add_subplot(121, projection='3d')

            # Plot context point cloud
            if laser_scan.has_colors():
                context_colors = np.asarray(laser_scan.colors)[context_idx]
                ax_context.scatter(context_pts[:,0], context_pts[:,1], context_pts[:,2],
                                 s=0.5, c=context_colors, alpha=0.4)
            else:
                ax_context.scatter(context_pts[:,0], context_pts[:,1], context_pts[:,2],
                                 s=0.5, c=context_pts[:,2], cmap='viridis', alpha=0.4)

            # Draw all objects in context (faded)
            for j, obj in enumerate(objects):
                obj_center = np.array(obj['center_scenefun3d'])
                obj_size = np.array(obj.get('size_m', obj.get('size', [1, 1, 1])))
                obj_color = object_colors[j % len(object_colors)]

                half_size = obj_size / 2
                corners = []
                for dx in [-1, 1]:
                    for dy in [-1, 1]:
                        for dz in [-1, 1]:
                            corner = obj_center + half_size * np.array([dx, dy, dz])
                            corners.append(corner)

                corners = np.array(corners)
                edges = [
                    (0, 1), (2, 3), (4, 5), (6, 7),
                    (0, 2), (1, 3), (4, 6), (5, 7),
                    (0, 4), (1, 5), (2, 6), (3, 7)
                ]

                for edge in edges:
                    points = corners[[edge[0], edge[1]]]
                    ax_context.plot3D(points[:, 0], points[:, 1], points[:, 2],
                                    color=obj_color, alpha=0.3, linewidth=2)

                ax_context.scatter(obj_center[0], obj_center[1], obj_center[2],
                                 c=obj_color, s=100, alpha=0.3, marker='o')

            # Draw all affordances with current one highlighted
            for j, other_aff in enumerate(affordances):
                if 'min_coords' in other_aff and 'max_coords' in other_aff:
                    other_min_coords = np.array(other_aff['min_coords'])
                    other_max_coords = np.array(other_aff['max_coords'])
                    other_center = np.array(other_aff['center'])

                    # Highlight current affordance
                    alpha = 1.0 if j == i else 0.3
                    linewidth = 4 if j == i else 1
                    marker_size = 300 if j == i else 50

                    corners = []
                    for dx in [other_min_coords[0], other_max_coords[0]]:
                        for dy in [other_min_coords[1], other_max_coords[1]]:
                            for dz in [other_min_coords[2], other_max_coords[2]]:
                                corners.append([dx, dy, dz])

                    corners = np.array(corners)
                    for edge in edges:
                        points = corners[[edge[0], edge[1]]]
                        ax_context.plot3D(points[:, 0], points[:, 1], points[:, 2],
                                        color='purple', alpha=alpha, linewidth=linewidth, linestyle='--')

                    ax_context.scatter(other_center[0], other_center[1], other_center[2],
                                     c='purple', s=marker_size, alpha=alpha, marker='^',
                                     edgecolors='white', linewidth=1)

            # Add ROI highlight box
            roi_padding = np.maximum(size * 1.5, [0.3, 0.3, 0.3])
            roi_min = center - roi_padding
            roi_max = center + roi_padding

            roi_corners = []
            for dx in [roi_min[0], roi_max[0]]:
                for dy in [roi_min[1], roi_max[1]]:
                    for dz in [roi_min[2], roi_max[2]]:
                        roi_corners.append([dx, dy, dz])

            roi_corners = np.array(roi_corners)
            for edge in edges:
                points = roi_corners[[edge[0], edge[1]]]
                ax_context.plot3D(points[:, 0], points[:, 1], points[:, 2],
                                color='lime', alpha=0.8, linewidth=3, linestyle=':')

            # Context view settings
            task_name = aff.get('task_description', 'Unknown task')
            ax_context.set_title(f'CONTEXT: Affordance "{task_name[:25]}..." Location\n(Green box shows zoomed region)',
                               fontsize=14, weight='bold')
            ax_context.set_xlabel('X (meters)', fontsize=10)
            ax_context.set_ylabel('Y (meters)', fontsize=10)
            ax_context.set_zlabel('Z (meters)', fontsize=10)

            # Set context limits
            all_points = context_pts
            max_range = np.array([
                np.max(all_points[:, j]) - np.min(all_points[:, j])
                for j in range(3)
            ]).max() / 2.0

            mid_x = np.mean(all_points[:, 0])
            mid_y = np.mean(all_points[:, 1])
            mid_z = np.mean(all_points[:, 2])

            ax_context.set_xlim(mid_x - max_range, mid_x + max_range)
            ax_context.set_ylim(mid_y - max_range, mid_y + max_range)
            ax_context.set_zlim(mid_z - max_range, mid_z + max_range)

            # === RIGHT PANEL: ZOOMED VIEW ===
            ax_zoom = fig.add_subplot(122, projection='3d')

            # Get ROI points
            roi_padding = np.maximum(size * 1.5, [0.3, 0.3, 0.3])
            min_roi = center - roi_padding
            max_roi = center + roi_padding

            mask = ((pts >= min_roi) & (pts <= max_roi)).all(axis=1)
            roi_pts = pts[mask]

            if len(roi_pts) < 50:
                roi_padding *= 2
                min_roi = center - roi_padding
                max_roi = center + roi_padding
                mask = ((pts >= min_roi) & (pts <= max_roi)).all(axis=1)
                roi_pts = pts[mask]

            if len(roi_pts) > 10000:
                roi_idx = np.random.choice(len(roi_pts), size=10000, replace=False)
                roi_pts = roi_pts[roi_idx]

            # Plot ROI point cloud
            if len(roi_pts) > 0:
                ax_zoom.scatter(roi_pts[:,0], roi_pts[:,1], roi_pts[:,2],
                               s=3, c=roi_pts[:,2], cmap='viridis', alpha=0.8)

            # Draw current affordance bounding box
            corners = []
            for dx in [min_coords[0], max_coords[0]]:
                for dy in [min_coords[1], max_coords[1]]:
                    for dz in [min_coords[2], max_coords[2]]:
                        corners.append([dx, dy, dz])

            corners = np.array(corners)
            for edge in edges:
                points = corners[[edge[0], edge[1]]]
                ax_zoom.plot3D(points[:, 0], points[:, 1], points[:, 2],
                              color='purple', alpha=0.9, linewidth=5, linestyle='-')

            # Add affordance center marker
            ax_zoom.scatter(center[0], center[1], center[2],
                           c='purple', s=500, alpha=1.0, marker='^',
                           edgecolors='white', linewidth=3)

            # Zoom view settings
            elev, azim = calculate_best_view_angle(center, size)
            ax_zoom.view_init(elev=elev, azim=azim)

            ax_zoom.set_title(f'ZOOM: Affordance Detail View\nTask: {task_name}\nPoints: {aff["point_count"]} | Size: [{size[0]:.3f}m × {size[1]:.3f}m × {size[2]:.3f}m]',
                            fontsize=14, weight='bold')
            ax_zoom.set_xlabel('X (meters)', fontsize=10)
            ax_zoom.set_ylabel('Y (meters)', fontsize=10)
            ax_zoom.set_zlabel('Z (meters)', fontsize=10)

            # Set zoom limits
            margin = np.maximum(size.max() * 0.8, 0.2)
            ax_zoom.set_xlim(center[0] - margin, center[0] + margin)
            ax_zoom.set_ylim(center[1] - margin, center[1] + margin)
            ax_zoom.set_zlim(center[2] - margin, center[2] + margin)

            # Add overall title
            fig.suptitle(f'Side-by-Side View: Affordance {i+1} - {task_name[:30]}... (ID: {aff["id"][:8]})',
                        fontsize=16, weight='bold', y=0.95)

            plt.tight_layout()

            # Save side-by-side affordance visualization
            task_safe = task_name.lower().replace(' ', '_').replace('/', '_')[:20]
            aff_filename = f"sidebyside_affordance_{i+1}_{task_safe}_{aff['id'][:8]}.png"
            output_file = affordances_dir / aff_filename
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"   🎯🔍 Affordance {i+1} ({task_name[:20]}...) side-by-side saved to: {aff_filename}")
            plt.close()

        print(f"✅ Side-by-side visualizations created: {len(objects)} objects, {len(affordances)} affordances")

    except Exception as e:
        print(f"❌ Error creating side-by-side visualizations: {e}")
        import traceback
        traceback.print_exc()

def create_labeled_point_cloud_visualizations(results, output_dir, results_file):
    """Create labeled point cloud visualizations."""
    if create_labeled_point_clouds is None:
        print("⚠️ Skipping labeled point cloud visualization - labeler not available")
        return

    try:
        print("\n🏷️ Creating labeled point cloud visualizations...")
        create_labeled_point_clouds(results_file, output_dir)

    except Exception as e:
        print(f"❌ Error creating labeled point cloud visualizations: {e}")
        import traceback
        traceback.print_exc()

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
        create_point_cloud_visualization(results, output_dir)  # New enhanced visualization
        create_individual_visualizations(results, output_dir)  # Individual focused visualizations
        create_side_by_side_visualizations(results, output_dir)  # Side-by-side context + zoom views
        create_labeled_point_cloud_visualizations(results, output_dir, results_file)  # Labeled PLY files
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