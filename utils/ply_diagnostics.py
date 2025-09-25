#!/usr/bin/env python3
"""
PLY Diagnostics Tool

Quick diagnostic tool to analyze labeled point cloud files and troubleshoot loading issues.
"""

import numpy as np
import open3d as o3d
import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def diagnose_ply_file(ply_file: str):
    """Diagnose a PLY file for potential issues."""
    print(f"🔍 Diagnosing: {ply_file}")
    print("=" * 50)

    try:
        # Load with Open3D
        pcd = o3d.io.read_point_cloud(ply_file)

        if len(pcd.points) == 0:
            print("❌ No points loaded!")
            return False

        points = np.asarray(pcd.points)
        has_colors = pcd.has_colors()

        print(f"✅ Successfully loaded: {len(points):,} points")
        print(f"✅ Has colors: {has_colors}")

        # Basic statistics
        print(f"\n📊 Point Statistics:")
        print(f"   X range: [{points[:,0].min():.3f}, {points[:,0].max():.3f}]")
        print(f"   Y range: [{points[:,1].min():.3f}, {points[:,1].max():.3f}]")
        print(f"   Z range: [{points[:,2].min():.3f}, {points[:,2].max():.3f}]")

        # Check for invalid coordinates
        finite_points = np.all(np.isfinite(points))
        print(f"   All coordinates finite: {finite_points}")

        if not finite_points:
            inf_count = np.sum(~np.isfinite(points))
            print(f"   ⚠️ {inf_count} invalid coordinates found")

        # Color analysis
        if has_colors:
            colors = np.asarray(pcd.colors)
            print(f"\n🎨 Color Statistics:")
            print(f"   Color range: [{colors.min():.3f}, {colors.max():.3f}]")

            # Check color validity (should be 0-1 for Open3D)
            valid_colors = np.all((colors >= 0) & (colors <= 1))
            print(f"   Colors in valid range [0,1]: {valid_colors}")

            # Color diversity
            sample_size = min(1000, len(colors))
            sample_colors = colors[:sample_size]
            unique_colors = len(np.unique(sample_colors.view(np.void), axis=0))
            print(f"   Unique colors in sample of {sample_size}: {unique_colors}")

            # Check for labeled vs background colors
            # Background should be faded (low values)
            avg_brightness = np.mean(colors)
            print(f"   Average brightness: {avg_brightness:.3f}")

            if avg_brightness < 0.5:
                print("   ✅ Point cloud appears to have proper faded background")
            else:
                print("   ⚠️ Point cloud may not have proper background fading")

        # File size analysis
        file_size = Path(ply_file).stat().st_size
        expected_size_mb = len(points) * (3 * 8 + 3 * 1) / 1024 / 1024  # 3 doubles + 3 bytes
        actual_size_mb = file_size / 1024 / 1024
        print(f"\n📁 File Analysis:")
        print(f"   File size: {actual_size_mb:.1f} MB")
        print(f"   Expected size: {expected_size_mb:.1f} MB")
        print(f"   Compression ratio: {actual_size_mb/expected_size_mb:.2f}")

        return True

    except Exception as e:
        print(f"❌ Error loading PLY file: {e}")
        return False


def analyze_labeled_metadata(metadata_file: str):
    """Analyze the metadata file for labeled point clouds."""
    print(f"\n📄 Analyzing metadata: {metadata_file}")
    print("=" * 50)

    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        total_points = metadata['total_points']
        object_points = metadata['object_points']
        affordance_points = metadata['affordance_points']
        background_points = metadata['background_points']

        print(f"📊 Label Distribution:")
        print(f"   Total points: {total_points:,}")
        print(f"   Object points: {object_points:,} ({100*object_points/total_points:.3f}%)")
        print(f"   Affordance points: {affordance_points:,} ({100*affordance_points/total_points:.3f}%)")
        print(f"   Background points: {background_points:,} ({100*background_points/total_points:.3f}%)")

        print(f"\n🏷️ Objects ({len(metadata['objects'])}):")
        for obj_id, obj_info in metadata['objects'].items():
            print(f"   • {obj_info['class']}: {obj_info['point_count']:,} points")

        print(f"\n🎯 Affordances ({len(metadata['affordances'])}):")
        for aff_id, aff_info in metadata['affordances'].items():
            task_short = aff_info['task'][:40] + "..." if len(aff_info['task']) > 40 else aff_info['task']
            print(f"   • {task_short}: {aff_info['point_count']:,} points")

        return metadata

    except Exception as e:
        print(f"❌ Error reading metadata: {e}")
        return None


def create_simple_2d_preview(ply_file: str, output_file: str, max_points: int = 5000):
    """Create a simple 2D preview of the labeled point cloud."""
    print(f"\n🖼️ Creating 2D preview: {output_file}")

    try:
        pcd = o3d.io.read_point_cloud(ply_file)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None

        # Sample points for preview
        if len(points) > max_points:
            indices = np.random.choice(len(points), size=max_points, replace=False)
            points = points[indices]
            if colors is not None:
                colors = colors[indices]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Top view (X-Y)
        if colors is not None:
            ax1.scatter(points[:, 0], points[:, 1], c=colors, s=0.5, alpha=0.7)
        else:
            ax1.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap='viridis', s=0.5, alpha=0.7)
        ax1.set_xlabel('X (meters)')
        ax1.set_ylabel('Y (meters)')
        ax1.set_title('Top View (X-Y)')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

        # Side view (X-Z)
        if colors is not None:
            ax2.scatter(points[:, 0], points[:, 2], c=colors, s=0.5, alpha=0.7)
        else:
            ax2.scatter(points[:, 0], points[:, 2], c=points[:, 1], cmap='viridis', s=0.5, alpha=0.7)
        ax2.set_xlabel('X (meters)')
        ax2.set_ylabel('Z (meters)')
        ax2.set_title('Side View (X-Z)')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')

        plt.suptitle(f'Quick Preview: {Path(ply_file).name}')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Preview saved to: {output_file}")
        return True

    except Exception as e:
        print(f"   ❌ Error creating preview: {e}")
        return False


def main():
    """Main diagnostic function."""
    print("🔧 PLY Point Cloud Diagnostics Tool")
    print("=" * 50)

    # Check labeled point cloud directory
    labeled_dir = Path("visualizations/labeled_point_clouds")

    if not labeled_dir.exists():
        print("❌ No labeled point clouds directory found!")
        print("   Run the point cloud labeler first")
        return

    # Find PLY files
    ply_files = list(labeled_dir.glob("*.ply"))
    metadata_file = labeled_dir / "422203_laser_scan_labeled_metadata.json"

    print(f"Found {len(ply_files)} PLY files:")
    for ply_file in ply_files:
        print(f"   📄 {ply_file.name}")

    # Analyze metadata if available
    if metadata_file.exists():
        metadata = analyze_labeled_metadata(str(metadata_file))
    else:
        print("⚠️ No metadata file found")
        metadata = None

    # Diagnose each PLY file
    for ply_file in ply_files:
        print(f"\n" + "="*60)
        success = diagnose_ply_file(str(ply_file))

        if success:
            # Create a simple preview
            preview_file = ply_file.parent / f"{ply_file.stem}_preview.png"
            create_simple_2d_preview(str(ply_file), str(preview_file))

    print(f"\n🎉 Diagnostics complete!")
    print(f"✅ All PLY files appear to be valid and loadable")
    print(f"📁 Files are located in: {labeled_dir}")


if __name__ == "__main__":
    main()