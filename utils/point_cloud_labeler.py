#!/usr/bin/env python3
"""
Point Cloud Labeler

Creates labeled point clouds by overlaying semantic information (objects and affordances)
onto the original laser scan data.
"""

import numpy as np
import open3d as o3d
import json
import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Add SceneFun3D to path
scenefun3d_root = Path("/home/jiachen/scratch/SceneFun3D/scenefun3d")
original_cwd = os.getcwd()
os.chdir(str(scenefun3d_root))
sys.path.insert(0, str(scenefun3d_root))

try:
    from utils.data_parser import DataParser
except ImportError:
    print("Warning: Could not import DataParser - point cloud loading may be limited")
    DataParser = None
finally:
    os.chdir(original_cwd)


class PointCloudLabeler:
    """Creates labeled point clouds with semantic information."""

    def __init__(self, data_root: str, visit_id: str):
        """Initialize the labeler.

        Args:
            data_root: Path to SceneFun3D data root
            visit_id: Visit ID for the scene
        """
        self.data_root = data_root
        self.visit_id = visit_id
        self.parser = DataParser(data_root) if DataParser else None

        # Color schemes
        self.object_colors = {
            'bathtub': [255, 50, 50],      # Red
            'sink': [255, 165, 0],         # Orange
            'toilet': [255, 255, 0],       # Yellow
            'chair': [0, 255, 255],        # Cyan
            'table': [255, 0, 255],        # Magenta
            'bed': [0, 255, 0],            # Green
            'sofa': [0, 0, 255],           # Blue
        }

        self.affordance_colors = {
            'close': [128, 0, 128],        # Purple
            'open': [148, 0, 211],         # Dark Violet
            'sit': [138, 43, 226],         # Blue Violet
            'place': [75, 0, 130],         # Indigo
            'pick': [106, 90, 205],        # Slate Blue
        }

        self.background_color = [200, 200, 200]  # Light gray

    def load_point_cloud(self) -> Tuple[np.ndarray, np.ndarray, o3d.geometry.PointCloud]:
        """Load the original point cloud data.

        Returns:
            Tuple of (points, colors, open3d_cloud)
        """
        if self.parser is None:
            raise ValueError("DataParser not available - cannot load point cloud")

        # Load laser scan
        laser_scan = self.parser.get_laser_scan(self.visit_id)
        points = np.asarray(laser_scan.points)

        # Get colors if available
        if laser_scan.has_colors():
            colors = np.asarray(laser_scan.colors)
            # Convert from [0,1] to [0,255] if needed
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
        else:
            # Create grayscale colors based on Z coordinate
            z_values = points[:, 2]
            z_normalized = (z_values - z_values.min()) / (z_values.max() - z_values.min())
            colors = np.stack([z_normalized, z_normalized, z_normalized], axis=1)
            colors = (colors * 255).astype(np.uint8)

        return points, colors, laser_scan

    def get_object_color(self, obj_class: str) -> List[int]:
        """Get color for object class."""
        return self.object_colors.get(obj_class.lower(), [100, 100, 100])

    def get_affordance_color(self, task_description: str) -> List[int]:
        """Get color for affordance based on task."""
        task_lower = task_description.lower()
        for keyword, color in self.affordance_colors.items():
            if keyword in task_lower:
                return color
        return [128, 0, 128]  # Default purple

    def points_in_bbox(self, points: np.ndarray, center: np.ndarray, size: np.ndarray,
                      padding_factor: float = 3.0) -> np.ndarray:
        """Find points within a bounding box with padding.

        Args:
            points: Point cloud array (N, 3)
            center: Bounding box center (3,)
            size: Bounding box size (3,)
            padding_factor: Expand bbox by this factor (default 3.0 for reasonable coverage)

        Returns:
            Boolean mask of points within bbox
        """
        # Apply padding to account for potentially undersized bounding boxes
        padded_size = np.array(size) * padding_factor

        # Ensure minimum size (at least 20cm in each dimension for objects)
        min_size = np.array([0.2, 0.2, 0.2])  # 20cm minimum
        padded_size = np.maximum(padded_size, min_size)

        half_size = padded_size / 2
        min_bounds = np.array(center) - half_size
        max_bounds = np.array(center) + half_size

        # Check if points are within bounds
        within_bounds = np.all((points >= min_bounds) & (points <= max_bounds), axis=1)
        return within_bounds

    def create_labeled_point_cloud(self, results: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Create a labeled point cloud with semantic colors.

        Args:
            results: Pipeline results containing objects and affordances

        Returns:
            Tuple of (points, labeled_colors, label_info)
        """
        print("🏷️ Creating labeled point cloud...")

        # Load original point cloud
        points, original_colors, _ = self.load_point_cloud()
        print(f"   Loaded {len(points):,} points")

        # Initialize with background colors (faded original)
        labeled_colors = (original_colors * 0.3).astype(np.uint8)  # Fade to 30%

        # Track label assignments
        label_info = {
            'total_points': int(len(points)),
            'background_points': int(len(points)),
            'object_points': 0,
            'affordance_points': 0,
            'objects': {},
            'affordances': {}
        }

        # Label object points
        objects = results.get('objects', [])
        for i, obj in enumerate(objects):
            center = np.array(obj['center_scenefun3d'])
            size = np.array(obj.get('size_m', obj.get('size', [1, 1, 1])))
            obj_class = obj['class']
            obj_id = obj['id']

            # Find points within object bounding box
            bbox_mask = self.points_in_bbox(points, center, size)
            obj_point_count = np.sum(bbox_mask)

            if obj_point_count > 0:
                # Color object points
                obj_color = self.get_object_color(obj_class)
                labeled_colors[bbox_mask] = obj_color

                # Update label info
                label_info['objects'][obj_id] = {
                    'class': obj_class,
                    'color': [int(c) for c in obj_color],
                    'point_count': int(obj_point_count),
                    'center': [float(x) for x in center.tolist()],
                    'size': [float(x) for x in size.tolist()]
                }
                label_info['object_points'] += int(obj_point_count)
                label_info['background_points'] -= int(obj_point_count)

                print(f"   📦 {obj_class}: {obj_point_count:,} points labeled")

        # Label affordance points (using annotation indices)
        affordances = results.get('affordances', [])

        # Load annotations to get point indices
        try:
            annotations = self.parser.get_annotations(self.visit_id)
            annotation_lookup = {ann['annot_id']: ann for ann in annotations}
        except:
            print("   ⚠️ Could not load annotations - skipping affordance point labeling")
            annotation_lookup = {}

        for i, aff in enumerate(affordances):
            aff_id = aff['id']
            task_description = aff.get('task_description', 'Unknown task')

            # Get point indices from annotation
            if aff_id in annotation_lookup:
                annotation = annotation_lookup[aff_id]
                indices = annotation.get('indices', [])

                if indices:
                    # Ensure indices are valid
                    valid_indices = [idx for idx in indices if 0 <= idx < len(points)]
                    aff_point_count = len(valid_indices)

                    if aff_point_count > 0:
                        # Color affordance points (blend with any existing object colors)
                        aff_color = self.get_affordance_color(task_description)

                        # Blend affordance color with existing colors for overlap visualization
                        existing_colors = labeled_colors[valid_indices]
                        blended_colors = (existing_colors * 0.5 + np.array(aff_color) * 0.5).astype(np.uint8)
                        labeled_colors[valid_indices] = blended_colors

                        # Update label info
                        label_info['affordances'][aff_id] = {
                            'task': task_description,
                            'color': [int(c) for c in aff_color],
                            'point_count': int(aff_point_count),
                            'center': [float(x) for x in aff['center']],
                            'indices': [int(idx) for idx in valid_indices[:100]]  # Store first 100 indices for reference
                        }
                        label_info['affordance_points'] += int(aff_point_count)

                        print(f"   🎯 {task_description[:30]}: {aff_point_count:,} points labeled")

        print(f"   ✅ Labeling complete: {label_info['object_points']:,} object points, " +
              f"{label_info['affordance_points']:,} affordance points, " +
              f"{label_info['background_points']:,} background points")

        return points, labeled_colors, label_info

    def save_labeled_ply(self, points: np.ndarray, colors: np.ndarray, output_file: str,
                        label_info: Optional[Dict] = None):
        """Save labeled point cloud as PLY file.

        Args:
            points: Point coordinates (N, 3)
            colors: Point colors (N, 3)
            output_file: Output PLY file path
            label_info: Optional label information for metadata
        """
        print(f"💾 Saving labeled PLY to: {output_file}")

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Open3D expects [0,1]

        # Save PLY file
        success = o3d.io.write_point_cloud(output_file, pcd)
        if success:
            print(f"   ✅ Saved {len(points):,} labeled points")
        else:
            print(f"   ❌ Failed to save PLY file")

        # Save metadata JSON if provided
        if label_info:
            metadata_file = output_file.replace('.ply', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(label_info, f, indent=2)
            print(f"   📄 Metadata saved to: {metadata_file}")

    def create_filtered_point_clouds(self, points: np.ndarray, colors: np.ndarray,
                                   label_info: Dict[str, Any], output_dir: Path):
        """Create filtered point clouds (objects only, affordances only).

        Args:
            points: All point coordinates
            colors: All point colors
            label_info: Label information
            output_dir: Output directory
        """
        print("🔍 Creating filtered point clouds...")

        # Objects only point cloud
        if label_info['objects']:
            object_mask = np.zeros(len(points), dtype=bool)

            # Mark all object points
            for obj_id, obj_info in label_info['objects'].items():
                center = np.array(obj_info['center'], dtype=float)
                size = np.array(obj_info['size'], dtype=float)
                bbox_mask = self.points_in_bbox(points, center, size, padding_factor=3.0)
                object_mask |= bbox_mask

            if np.sum(object_mask) > 0:
                objects_file = output_dir / f"{self.visit_id}_laser_scan_objects_only.ply"
                self.save_labeled_ply(points[object_mask], colors[object_mask], str(objects_file))

        # Affordances only point cloud
        if label_info['affordances']:
            affordance_mask = np.zeros(len(points), dtype=bool)

            # Mark all affordance points using stored indices
            for aff_id, aff_info in label_info['affordances'].items():
                if 'indices' in aff_info:
                    # Get all indices (not just the stored subset)
                    try:
                        annotations = self.parser.get_annotations(self.visit_id)
                        annotation = next((ann for ann in annotations if ann['annot_id'] == aff_id), None)
                        if annotation and 'indices' in annotation:
                            indices = annotation['indices']
                            valid_indices = [idx for idx in indices if 0 <= idx < len(points)]
                            affordance_mask[valid_indices] = True
                    except:
                        pass

            if np.sum(affordance_mask) > 0:
                affordances_file = output_dir / f"{self.visit_id}_laser_scan_affordances_only.ply"
                self.save_labeled_ply(points[affordance_mask], colors[affordance_mask], str(affordances_file))

    def create_legend(self, label_info: Dict[str, Any], output_file: str):
        """Create a color legend for the labeled point cloud.

        Args:
            label_info: Label information
            output_file: Output image file for legend
        """
        print(f"🎨 Creating color legend...")

        fig, ax = plt.subplots(figsize=(10, 8))

        # Prepare legend items
        legend_items = []
        y_pos = 0.9

        # Add title
        ax.text(0.5, 0.95, 'Point Cloud Label Legend', ha='center', va='top',
               fontsize=16, weight='bold', transform=ax.transAxes)

        # Objects section
        if label_info['objects']:
            ax.text(0.05, y_pos, 'OBJECTS:', ha='left', va='top', fontsize=14,
                   weight='bold', transform=ax.transAxes)
            y_pos -= 0.06

            for obj_id, obj_info in label_info['objects'].items():
                color = np.array(obj_info['color']) / 255.0
                obj_class = obj_info['class']
                point_count = obj_info['point_count']

                # Draw color patch
                rect = plt.Rectangle((0.1, y_pos - 0.02), 0.03, 0.03,
                                   facecolor=color, edgecolor='black', transform=ax.transAxes)
                ax.add_patch(rect)

                # Add label
                ax.text(0.15, y_pos, f"{obj_class.title()} ({point_count:,} points)",
                       ha='left', va='center', fontsize=12, transform=ax.transAxes)
                y_pos -= 0.05

        # Affordances section
        y_pos -= 0.03
        if label_info['affordances']:
            ax.text(0.05, y_pos, 'AFFORDANCES:', ha='left', va='top', fontsize=14,
                   weight='bold', transform=ax.transAxes)
            y_pos -= 0.06

            for aff_id, aff_info in label_info['affordances'].items():
                color = np.array(aff_info['color']) / 255.0
                task = aff_info['task']
                point_count = aff_info['point_count']

                # Draw color patch
                rect = plt.Rectangle((0.1, y_pos - 0.02), 0.03, 0.03,
                                   facecolor=color, edgecolor='black', transform=ax.transAxes)
                ax.add_patch(rect)

                # Add label (truncate long task descriptions)
                task_short = task if len(task) <= 40 else task[:37] + "..."
                ax.text(0.15, y_pos, f"{task_short} ({point_count:,} points)",
                       ha='left', va='center', fontsize=10, transform=ax.transAxes)
                y_pos -= 0.04

        # Background section
        y_pos -= 0.03
        color = np.array(self.background_color) / 255.0
        bg_count = label_info['background_points']

        ax.text(0.05, y_pos, 'BACKGROUND:', ha='left', va='top', fontsize=14,
               weight='bold', transform=ax.transAxes)
        y_pos -= 0.06

        rect = plt.Rectangle((0.1, y_pos - 0.02), 0.03, 0.03,
                           facecolor=color, edgecolor='black', transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.15, y_pos, f"Unlabeled points ({bg_count:,} points)",
               ha='left', va='center', fontsize=12, transform=ax.transAxes)

        # Statistics
        y_pos -= 0.1
        total_points = label_info['total_points']
        ax.text(0.05, y_pos, 'STATISTICS:', ha='left', va='top', fontsize=14,
               weight='bold', transform=ax.transAxes)
        y_pos -= 0.05
        ax.text(0.1, y_pos, f"Total Points: {total_points:,}", ha='left', va='center',
               fontsize=11, transform=ax.transAxes)
        y_pos -= 0.04
        ax.text(0.1, y_pos, f"Labeled Points: {total_points - bg_count:,} " +
               f"({100 * (total_points - bg_count) / total_points:.1f}%)",
               ha='left', va='center', fontsize=11, transform=ax.transAxes)

        # Remove axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Save legend
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   🎨 Legend saved to: {output_file}")

    def create_2d_validation_plots(self, points: np.ndarray, colors: np.ndarray,
                                 label_info: Dict[str, Any], output_dir: Path):
        """Create 2D validation plots to check labeled point clouds.

        Args:
            points: Point coordinates (N, 3)
            colors: Point colors (N, 3)
            label_info: Label information
            output_dir: Output directory
        """
        print(f"📊 Creating 2D validation plots...")

        # Sample points for visualization (max 5k for faster rendering)
        n_sample = min(5000, len(points))
        indices = np.random.choice(len(points), size=n_sample, replace=False)
        sample_points = points[indices]
        sample_colors = colors[indices] / 255.0  # Normalize for matplotlib

        # Create figure with multiple 2D projections
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # X-Y projection (top view)
        ax1.scatter(sample_points[:, 0], sample_points[:, 1],
                   c=sample_colors, s=1, alpha=0.7)
        ax1.set_xlabel('X (meters)')
        ax1.set_ylabel('Y (meters)')
        ax1.set_title('Top View (X-Y Projection)')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

        # X-Z projection (front view)
        ax2.scatter(sample_points[:, 0], sample_points[:, 2],
                   c=sample_colors, s=1, alpha=0.7)
        ax2.set_xlabel('X (meters)')
        ax2.set_ylabel('Z (meters)')
        ax2.set_title('Front View (X-Z Projection)')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')

        # Y-Z projection (side view)
        ax3.scatter(sample_points[:, 1], sample_points[:, 2],
                   c=sample_colors, s=1, alpha=0.7)
        ax3.set_xlabel('Y (meters)')
        ax3.set_ylabel('Z (meters)')
        ax3.set_title('Side View (Y-Z Projection)')
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')

        # Point distribution by label
        ax4.axis('off')

        # Create custom legend showing actual colors used
        legend_elements = []

        # Add background
        bg_count = label_info['background_points']
        bg_color = np.array(self.background_color) / 255.0 * 0.3  # Faded
        legend_elements.append(plt.scatter([], [], c=[bg_color], s=100, alpha=0.7,
                                         label=f'Background ({bg_count:,} points)'))

        # Add objects
        for obj_id, obj_info in label_info.get('objects', {}).items():
            obj_color = np.array(obj_info['color']) / 255.0
            obj_class = obj_info['class']
            obj_count = obj_info['point_count']
            legend_elements.append(plt.scatter([], [], c=[obj_color], s=100, alpha=0.9,
                                             label=f'{obj_class.title()} ({obj_count:,} points)'))

        # Add affordances
        for aff_id, aff_info in label_info.get('affordances', {}).items():
            aff_color = np.array(aff_info['color']) / 255.0
            aff_task = aff_info['task']
            aff_count = aff_info['point_count']
            task_short = aff_task if len(aff_task) <= 25 else aff_task[:22] + "..."
            legend_elements.append(plt.scatter([], [], c=[aff_color], s=100, alpha=0.9,
                                             label=f'{task_short} ({aff_count:,} points)'))

        ax4.legend(handles=legend_elements, loc='center', fontsize=10)
        ax4.set_title('Label Distribution\n(Color codes used in point cloud)')

        # Add statistics
        stats_text = f"""
Statistics:
• Total Points: {label_info['total_points']:,}
• Sampled: {n_sample:,}
• Objects: {len(label_info.get('objects', {}))} ({label_info['object_points']:,} points)
• Affordances: {len(label_info.get('affordances', {}))} ({label_info['affordance_points']:,} points)
• Background: {label_info['background_points']:,} points
• Coverage: {100 * (label_info['total_points'] - label_info['background_points']) / label_info['total_points']:.1f}%
        """
        ax4.text(0.02, 0.02, stats_text.strip(), transform=ax4.transAxes,
                fontsize=9, verticalalignment='bottom', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

        plt.suptitle(f'2D Validation: Labeled Point Cloud - Visit {self.visit_id}',
                    fontsize=16, weight='bold')
        plt.tight_layout()

        # Save validation plot
        output_file = output_dir / '2d_validation_labeled_points.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   📊 2D validation plot saved to: {output_file.name}")

    def validate_ply_files(self, output_dir: Path):
        """Validate generated PLY files by loading them back.

        Args:
            output_dir: Directory containing PLY files
        """
        print(f"✅ Validating PLY files...")

        ply_files = list(output_dir.glob("*.ply"))
        if not ply_files:
            print("   ⚠️ No PLY files found to validate")
            return

        validation_results = {}

        for ply_file in ply_files:
            try:
                # Try to load with Open3D
                pcd = o3d.io.read_point_cloud(str(ply_file))

                if len(pcd.points) == 0:
                    status = "❌ FAILED - No points loaded"
                else:
                    points = np.asarray(pcd.points)
                    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

                    # Basic validation checks
                    has_colors = colors is not None
                    finite_points = np.all(np.isfinite(points))
                    valid_colors = True
                    if has_colors:
                        valid_colors = np.all((colors >= 0) & (colors <= 1))

                    if finite_points and valid_colors:
                        status = f"✅ VALID - {len(points):,} points, colors: {has_colors}"
                    else:
                        issues = []
                        if not finite_points:
                            issues.append("infinite coordinates")
                        if not valid_colors:
                            issues.append("invalid colors")
                        status = f"⚠️ ISSUES - {', '.join(issues)}"

                validation_results[ply_file.name] = {
                    'status': status,
                    'point_count': len(pcd.points),
                    'has_colors': pcd.has_colors()
                }

            except Exception as e:
                validation_results[ply_file.name] = {
                    'status': f"❌ ERROR - {str(e)[:50]}...",
                    'point_count': 0,
                    'has_colors': False
                }

        # Print validation results
        print("   PLY File Validation Results:")
        for filename, result in validation_results.items():
            print(f"   📄 {filename}: {result['status']}")

        return validation_results

    def create_debug_visualization(self, points: np.ndarray, results: Dict[str, Any], output_dir: Path):
        """Create debug visualization showing object and affordance locations.

        Args:
            points: Point coordinates (N, 3)
            results: Pipeline results
            output_dir: Output directory
        """
        print(f"🐛 Creating debug visualization...")

        # Sample points for performance (reduced for faster rendering)
        n_sample = min(8000, len(points))
        indices = np.random.choice(len(points), size=n_sample, replace=False)
        sample_points = points[indices]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Plot 1: Top view with object bounding boxes
        ax1.scatter(sample_points[:, 0], sample_points[:, 1], c='lightgray', s=0.5, alpha=0.5)

        objects = results['objects']
        object_colors = ['red', 'orange', 'yellow', 'cyan', 'magenta']

        for i, obj in enumerate(objects):
            center = np.array(obj['center_scenefun3d'])
            size = np.array(obj.get('size_m', obj.get('size', [1, 1, 1])))
            color = object_colors[i % len(object_colors)]

            # Draw both original and padded bounding boxes
            # Original bbox
            half_size = size / 2
            rect_orig = plt.Rectangle((center[0] - half_size[0], center[1] - half_size[1]),
                                    size[0], size[1], fill=False, edgecolor=color,
                                    linewidth=1, linestyle='--', alpha=0.5)
            ax1.add_patch(rect_orig)

            # Padded bbox (3x with 20cm minimum)
            padded_size = np.maximum(size * 3.0, [0.2, 0.2, 0.2])
            half_padded = padded_size / 2
            rect_padded = plt.Rectangle((center[0] - half_padded[0], center[1] - half_padded[1]),
                                      padded_size[0], padded_size[1], fill=False, edgecolor=color,
                                      linewidth=3, alpha=0.9)
            ax1.add_patch(rect_padded)

            # Mark center
            ax1.plot(center[0], center[1], 'o', color=color, markersize=8, markeredgecolor='white')

            # Add label
            ax1.annotate(f"{obj['class']}\n{obj['id'][:8]}\nOrig:{size[0]:.3f}×{size[1]:.3f}m\nPad:{padded_size[0]:.3f}×{padded_size[1]:.3f}m",
                        (center[0], center[1]), xytext=(10, 10), textcoords='offset points',
                        fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))

        ax1.set_xlabel('X (meters)')
        ax1.set_ylabel('Y (meters)')
        ax1.set_title('Top View: Objects with Original (--) and Padded (—) Bounding Boxes')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

        # Plot 2: Affordance locations
        ax2.scatter(sample_points[:, 0], sample_points[:, 1], c='lightgray', s=0.5, alpha=0.5)

        affordances = results['affordances']
        for i, aff in enumerate(affordances):
            center = np.array(aff['center'])

            # Draw affordance bounding box if available
            if 'min_coords' in aff and 'max_coords' in aff:
                min_coords = np.array(aff['min_coords'])
                max_coords = np.array(aff['max_coords'])
                size = max_coords - min_coords

                rect = plt.Rectangle((min_coords[0], min_coords[1]),
                                   size[0], size[1], fill=False, edgecolor='purple',
                                   linewidth=2, alpha=0.8)
                ax2.add_patch(rect)

            # Mark center
            ax2.plot(center[0], center[1], '^', color='purple', markersize=10, markeredgecolor='white')

            # Add label
            task_name = aff.get('task_description', 'Unknown')[:20]
            ax2.annotate(f"{task_name}\n{aff['id'][:8]}\n{aff['point_count']} pts",
                        (center[0], center[1]), xytext=(10, 10), textcoords='offset points',
                        fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='lavender',
                                            edgecolor='purple', alpha=0.8))

        ax2.set_xlabel('X (meters)')
        ax2.set_ylabel('Y (meters)')
        ax2.set_title('Top View: Affordances with Bounding Boxes')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')

        plt.suptitle(f'Debug Visualization: Object & Affordance Locations - Visit {self.visit_id}',
                    fontsize=16, weight='bold')
        plt.tight_layout()

        # Save debug plot
        output_file = output_dir / 'debug_object_affordance_locations.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   🐛 Debug visualization saved to: {output_file.name}")

    def create_interactive_html_viewer(self, points: np.ndarray, colors: np.ndarray,
                                     label_info: Dict[str, Any], output_file: str):
        """Create an interactive HTML viewer for the labeled point cloud.

        Args:
            points: Point coordinates (N, 3)
            colors: Point colors (N, 3)
            label_info: Label information
            output_file: Output HTML file path
        """
        print(f"🌐 Creating interactive HTML viewer...")

        # Sample points for web performance (max 15k points for faster loading)
        n_sample = min(15000, len(points))
        if n_sample < len(points):
            indices = np.random.choice(len(points), size=n_sample, replace=False)
            sample_points = points[indices]
            sample_colors = colors[indices]
            print(f"   Sampling {n_sample:,} points for faster web loading")
        else:
            sample_points = points
            sample_colors = colors

        # Create HTML content
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Labeled Point Cloud Viewer - Visit {self.visit_id}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ text-align: center; margin-bottom: 20px; }}
        .stats {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .legend {{ background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .color-box {{ display: inline-block; width: 20px; height: 20px; margin-right: 10px; border: 1px solid #ccc; }}
        .controls {{ margin: 20px 0; }}
        .control-group {{ margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Labeled Point Cloud Viewer</h1>
            <h2>Visit ID: {self.visit_id}</h2>
            <p>Interactive 3D visualization of semantically labeled point cloud</p>
        </div>

        <div class="controls">
            <div class="control-group">
                <label><input type="checkbox" id="showObjects" checked> Show Objects</label>
                <label style="margin-left: 20px;"><input type="checkbox" id="showAffordances" checked> Show Affordances</label>
                <label style="margin-left: 20px;"><input type="checkbox" id="showBackground" checked> Show Background</label>
            </div>
            <div class="control-group">
                <button onclick="resetView()">Reset View</button>
                <button onclick="topView()">Top View</button>
                <button onclick="sideView()">Side View</button>
            </div>
        </div>

        <div id="plotDiv" style="width:100%; height:600px;"></div>

        <div class="stats">
            <h3>Statistics</h3>
            <p><strong>Total Points:</strong> {label_info['total_points']:,}</p>
            <p><strong>Object Points:</strong> {label_info['object_points']:,}</p>
            <p><strong>Affordance Points:</strong> {label_info['affordance_points']:,}</p>
            <p><strong>Background Points:</strong> {label_info['background_points']:,}</p>
            <p><strong>Displayed Points:</strong> {n_sample:,}</p>
        </div>

        <div class="legend">
            <h3>Color Legend</h3>
            <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                <div>
                    <h4>Objects</h4>"""

        # Add object legend items
        for obj_id, obj_info in label_info.get('objects', {}).items():
            color = f"rgb({obj_info['color'][0]}, {obj_info['color'][1]}, {obj_info['color'][2]})"
            html_content += f"""
                    <div><span class="color-box" style="background-color: {color};"></span>{obj_info['class'].title()} ({obj_info['point_count']:,} points)</div>"""

        html_content += """
                </div>
                <div>
                    <h4>Affordances</h4>"""

        # Add affordance legend items
        for aff_id, aff_info in label_info.get('affordances', {}).items():
            color = f"rgb({aff_info['color'][0]}, {aff_info['color'][1]}, {aff_info['color'][2]})"
            task_short = aff_info['task'] if len(aff_info['task']) <= 30 else aff_info['task'][:27] + "..."
            html_content += f"""
                    <div><span class="color-box" style="background-color: {color};"></span>{task_short} ({aff_info['point_count']:,} points)</div>"""

        html_content += f"""
                </div>
            </div>
        </div>
    </div>

    <script>
        // Point cloud data
        var points = {{
            x: [{', '.join(map(str, sample_points[:, 0]))}],
            y: [{', '.join(map(str, sample_points[:, 1]))}],
            z: [{', '.join(map(str, sample_points[:, 2]))}],
            colors: ["""

        # Add color data
        for i, color in enumerate(sample_colors):
            rgb = f"rgb({color[0]}, {color[1]}, {color[2]})"
            html_content += f"'{rgb}'"
            if i < len(sample_colors) - 1:
                html_content += ", "

        html_content += f"""]
        }};

        // Create 3D scatter plot with performance optimizations
        var trace = {{
            x: points.x,
            y: points.y,
            z: points.z,
            mode: 'markers',
            type: 'scatter3d',
            marker: {{
                size: 1.5,
                color: points.colors,
                opacity: 0.7,
                line: {{width: 0}}  // Remove marker outlines for better performance
            }},
            hovertemplate: 'X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>',
            hoverinfo: 'x+y+z'  // Simplified hover info
        }};

        var layout = {{
            title: 'Labeled Point Cloud - Visit {self.visit_id} ({n_sample:,} points)',
            scene: {{
                xaxis: {{ title: 'X (meters)' }},
                yaxis: {{ title: 'Y (meters)' }},
                zaxis: {{ title: 'Z (meters)' }},
                camera: {{
                    eye: {{ x: 1.5, y: 1.5, z: 1.5 }}
                }}
            }},
            margin: {{ l: 0, r: 0, b: 0, t: 40 }},
            showlegend: false  // Disable legend for better performance
        }};

        var config = {{
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d', 'autoScale2d'],
            displaylogo: false,
            responsive: true
        }};

        Plotly.newPlot('plotDiv', [trace], layout, config);

        // Control functions
        function resetView() {{
            var update = {{
                'scene.camera.eye': {{ x: 1.5, y: 1.5, z: 1.5 }}
            }};
            Plotly.relayout('plotDiv', update);
        }}

        function topView() {{
            var update = {{
                'scene.camera.eye': {{ x: 0, y: 0, z: 2.5 }}
            }};
            Plotly.relayout('plotDiv', update);
        }}

        function sideView() {{
            var update = {{
                'scene.camera.eye': {{ x: 2.5, y: 0, z: 0 }}
            }};
            Plotly.relayout('plotDiv', update);
        }}

        // Toggle functionality would require more complex data management
        // For now, checkboxes are visual indicators

        console.log('Interactive point cloud viewer loaded successfully');
        console.log('Points displayed: {n_sample:,}');
        console.log('Objects: {len(label_info.get("objects", {}))}');
        console.log('Affordances: {len(label_info.get("affordances", {}))}');
    </script>
</body>
</html>"""

        # Save HTML file
        with open(output_file, 'w') as f:
            f.write(html_content)

        print(f"   🌐 Interactive viewer saved to: {output_file}")
        print(f"   📊 Displaying {n_sample:,} points")


def create_labeled_point_clouds(results_file: str, output_dir: Path,
                               data_root: str = "/home/jiachen/scratch/SceneFun3D/alignment/data_examples/scenefun3d"):
    """Main function to create labeled point clouds from pipeline results.

    Args:
        results_file: Path to pipeline results JSON
        output_dir: Output directory for labeled point clouds
        data_root: SceneFun3D data root directory
    """
    print("🏷️ Creating Labeled Point Clouds")
    print("=" * 50)

    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)

    visit_id = results.get('visit_id', '422203')
    print(f"Visit ID: {visit_id}")
    print(f"Objects: {len(results.get('objects', []))}")
    print(f"Affordances: {len(results.get('affordances', []))}")

    # Create output directory
    labeled_dir = output_dir / 'labeled_point_clouds'
    labeled_dir.mkdir(parents=True, exist_ok=True)

    # Initialize labeler
    labeler = PointCloudLabeler(data_root, visit_id)

    try:
        # Create labeled point cloud
        points, labeled_colors, label_info = labeler.create_labeled_point_cloud(results)

        # Save main labeled PLY
        main_ply_file = labeled_dir / f"{visit_id}_laser_scan_labeled.ply"
        labeler.save_labeled_ply(points, labeled_colors, str(main_ply_file), label_info)

        # Create filtered point clouds
        labeler.create_filtered_point_clouds(points, labeled_colors, label_info, labeled_dir)

        # Create validation and debugging visualizations
        print("\n🔍 Creating validation visualizations...")
        labeler.create_debug_visualization(points, results, labeled_dir)
        labeler.create_2d_validation_plots(points, labeled_colors, label_info, labeled_dir)

        # Validate PLY files
        validation_results = labeler.validate_ply_files(labeled_dir)

        # Create legend
        legend_file = labeled_dir / 'label_legend.png'
        labeler.create_legend(label_info, str(legend_file))

        # Create interactive HTML viewer
        html_file = labeled_dir / 'interactive_labeled_viewer.html'
        labeler.create_interactive_html_viewer(points, labeled_colors, label_info, str(html_file))

        print(f"\n🎉 Labeled point clouds created successfully!")
        print(f"📁 Output directory: {labeled_dir}")
        print("Generated files:")
        for file in labeled_dir.glob("*"):
            if file.is_file():
                print(f"   - {file.name}")

    except Exception as e:
        print(f"❌ Error creating labeled point clouds: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Test with current results
    results_file = "results/pipeline_results.json"
    output_dir = Path("visualizations")

    if Path(results_file).exists():
        create_labeled_point_clouds(results_file, output_dir)
    else:
        print(f"Results file not found: {results_file}")
        print("Run pipeline.py first")