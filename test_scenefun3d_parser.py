#!/usr/bin/env python3
"""
Test SceneFun3D DataParser functionality for compatibility with our pipeline.

This script tests the official SceneFun3D DataParser to ensure it works correctly
with our data and can replace our custom parsers.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add SceneFun3D paths
scenefun3d_root = Path(__file__).parent.parent / "scenefun3d"

# Change working directory to scenefun3d root to fix relative imports
original_cwd = os.getcwd()
os.chdir(str(scenefun3d_root))
sys.path.insert(0, str(scenefun3d_root))

try:
    from utils.data_parser import DataParser
    print("✅ Successfully imported SceneFun3D DataParser")
except ImportError as e:
    print(f"❌ Failed to import SceneFun3D DataParser: {e}")
    print(f"Current directory: {os.getcwd()}")
    sys.exit(1)
finally:
    # Restore original working directory
    os.chdir(original_cwd)

def test_data_parser():
    """Test SceneFun3D DataParser with our data."""

    # Initialize DataParser with correct data root - should point to where visit directories are
    # Based on data_parser_paths.py, it expects: <data_dir>/<visit_id>/<visit_id>_laser_scan.ply
    # Our structure is: data_examples/scenefun3d/visit_422203/422203_laser_scan.ply
    # So data_root should be: data_examples/scenefun3d/

    alignment_dir = Path(__file__).parent
    data_root = alignment_dir / "data_examples" / "scenefun3d"

    print(f"   📁 Using data root: {data_root}")
    print(f"   📁 Data root exists: {data_root.exists()}")

    parser = DataParser(str(data_root))

    visit_id = "422203"
    video_id = "42445781"

    print(f"\n🔧 Testing SceneFun3D DataParser with visit_id={visit_id}, video_id={video_id}")

    # Test 1: Load laser scan
    print("\n1. Testing get_laser_scan()...")
    try:
        laser_scan = parser.get_laser_scan(visit_id)
        print(f"   ✅ Loaded laser scan: {len(laser_scan.points):,} points")

        # Compare with our custom loader
        custom_scan_path = Path(__file__).parent / "data_examples/scenefun3d/visit_422203/422203_laser_scan.ply"
        if custom_scan_path.exists():
            import open3d as o3d
            custom_scan = o3d.io.read_point_cloud(str(custom_scan_path))
            print(f"   📊 Custom loader: {len(custom_scan.points):,} points")
            print(f"   📊 Point count match: {len(laser_scan.points) == len(custom_scan.points)}")

    except Exception as e:
        print(f"   ❌ Failed to load laser scan: {e}")

    # Test 2: Load ARKit reconstruction
    print("\n2. Testing get_arkit_reconstruction()...")
    try:
        arkit_pc = parser.get_arkit_reconstruction(visit_id, video_id, format="point_cloud")
        print(f"   ✅ Loaded ARKit point cloud: {len(arkit_pc.points):,} points")

        arkit_mesh = parser.get_arkit_reconstruction(visit_id, video_id, format="mesh")
        print(f"   ✅ Loaded ARKit mesh: {len(arkit_mesh.vertices):,} vertices, {len(arkit_mesh.triangles):,} faces")

    except Exception as e:
        print(f"   ❌ Failed to load ARKit reconstruction: {e}")

    # Test 3: Load transformation matrix
    print("\n3. Testing get_transform()...")
    try:
        transform = parser.get_transform(visit_id, video_id)
        print(f"   ✅ Loaded transform matrix: {transform.shape}")
        print(f"   📊 Transform:\n{transform}")

        # Compare with custom loader
        custom_transform_path = Path(__file__).parent / f"data_examples/scenefun3d/visit_422203/{video_id}/{video_id}_transform.npy"
        if custom_transform_path.exists():
            custom_transform = np.load(custom_transform_path)
            print(f"   📊 Custom transform: {custom_transform.shape}")
            print(f"   📊 Transform match: {np.allclose(transform, custom_transform)}")

    except Exception as e:
        print(f"   ❌ Failed to load transform: {e}")

    # Test 4: Load annotations
    print("\n4. Testing get_annotations()...")
    try:
        annotations = parser.get_annotations(visit_id)
        print(f"   ✅ Loaded annotations: {len(annotations)} annotations")

        # Print first annotation details
        if annotations:
            first_annot = annotations[0]
            print(f"   📊 First annotation ID: {first_annot['annot_id']}")
            print(f"   📊 First annotation indices count: {len(first_annot['indices'])}")
            print(f"   📊 First few indices: {first_annot['indices'][:5]}")

    except Exception as e:
        print(f"   ❌ Failed to load annotations: {e}")

    # Test 5: Load descriptions
    print("\n5. Testing get_descriptions()...")
    try:
        descriptions = parser.get_descriptions(visit_id)
        print(f"   ✅ Loaded descriptions: {len(descriptions)} task descriptions")

        for i, desc in enumerate(descriptions):
            print(f"   📋 Task {i+1}: {desc['description']}")

    except Exception as e:
        print(f"   ❌ Failed to load descriptions: {e}")

    # Test 6: Load motions
    print("\n6. Testing get_motions()...")
    try:
        motions = parser.get_motions(visit_id)
        print(f"   ✅ Loaded motions: {len(motions)} motion annotations")

        # Analyze motion types
        motion_types = [m['motion_type'] for m in motions]
        print(f"   📊 Motion types: {set(motion_types)}")

    except Exception as e:
        print(f"   ❌ Failed to load motions: {e}")

def test_coordinate_transformation():
    """Test coordinate transformation functionality."""

    print(f"\n🔄 Testing coordinate transformation...")

    data_root = Path(__file__).parent.parent
    parser = DataParser(str(data_root))

    visit_id = "422203"
    video_id = "42445781"

    try:
        # Load data
        laser_scan = parser.get_laser_scan(visit_id)
        arkit_pc = parser.get_arkit_reconstruction(visit_id, video_id, format="point_cloud")
        transform = parser.get_transform(visit_id, video_id)

        # Get a few sample points from laser scan
        laser_points = np.asarray(laser_scan.points)[:100]  # First 100 points
        arkit_points = np.asarray(arkit_pc.points)[:100]     # First 100 points

        print(f"   📊 Laser scan sample bounds: {laser_points.min(axis=0)} to {laser_points.max(axis=0)}")
        print(f"   📊 ARKit points sample bounds: {arkit_points.min(axis=0)} to {arkit_points.max(axis=0)}")

        # Test transformation (SceneFun3D → ARKitScenes)
        # Add homogeneous coordinates
        laser_homogeneous = np.hstack([laser_points, np.ones((len(laser_points), 1))])
        transformed_points = (laser_homogeneous @ transform.T)[:, :3]

        print(f"   📊 Transformed points bounds: {transformed_points.min(axis=0)} to {transformed_points.max(axis=0)}")

        # Check if transformation brings points into similar range as ARKit
        laser_range = laser_points.max(axis=0) - laser_points.min(axis=0)
        arkit_range = arkit_points.max(axis=0) - arkit_points.min(axis=0)
        transformed_range = transformed_points.max(axis=0) - transformed_points.min(axis=0)

        print(f"   📊 Coordinate ranges:")
        print(f"       Laser scan: {laser_range}")
        print(f"       ARKit points: {arkit_range}")
        print(f"       Transformed: {transformed_range}")

    except Exception as e:
        print(f"   ❌ Transformation test failed: {e}")

if __name__ == "__main__":
    print("🧪 SceneFun3D DataParser Compatibility Test")
    print("=" * 50)

    test_data_parser()
    test_coordinate_transformation()

    print("\n✨ Test completed!")