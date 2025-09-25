"""
Enhanced Coordinate Transformation Utilities

This module implements coordinate system inversion - transforming ARKitScenes data
to SceneFun3D coordinate system instead of the reverse.

Key Changes:
- Transform ARKit object bounding boxes, centroids, vertices to laser scan coordinates
- Keep affordance points in native SceneFun3D coordinate system (no transformation)
- Use inverse transformation matrix for ARKit → SceneFun3D conversion
"""

import numpy as np
import open3d as o3d
import os
from typing import Tuple, Optional, Union, List
from pathlib import Path
import sys

# Import from SceneFun3D toolkit
scenefun3d_root = Path(__file__).parent.parent.parent / "scenefun3d"

# Change working directory temporarily for imports
original_cwd = os.getcwd()
os.chdir(str(scenefun3d_root))
sys.path.insert(0, str(scenefun3d_root))

try:
    from utils.data_parser import DataParser
finally:
    os.chdir(original_cwd)


class EnhancedCoordinateTransformer:
    """Enhanced coordinate transformer using SceneFun3D toolkit with coordinate inversion."""

    def __init__(self, data_root: str, visit_id: str, video_id: str):
        """Initialize with SceneFun3D DataParser."""
        self.data_root = data_root
        self.visit_id = visit_id
        self.video_id = video_id

        # Initialize SceneFun3D DataParser
        self.parser = DataParser(data_root)

        # Load transformation matrix
        self.transform_matrix = self.parser.get_transform(visit_id, video_id)

        # Compute inverse transformation (ARKitScenes → SceneFun3D)
        self.inverse_transform = np.linalg.inv(self.transform_matrix)

        print(f"✅ Loaded transformation matrix: {self.transform_matrix.shape}")
        print(f"📊 Transform determinant: {np.linalg.det(self.transform_matrix):.6f}")

    def get_transformation_info(self) -> dict:
        """Get information about the transformation."""
        rotation_part = self.transform_matrix[:3, :3]
        translation_part = self.transform_matrix[:3, 3]

        # Extract rotation angle (assuming rotation around Z-axis)
        rotation_angle_rad = np.arctan2(rotation_part[1, 0], rotation_part[0, 0])
        rotation_angle_deg = np.degrees(rotation_angle_rad)

        return {
            'transform_matrix': self.transform_matrix,
            'inverse_transform': self.inverse_transform,
            'rotation_angle_deg': rotation_angle_deg,
            'translation': translation_part,
            'scale': np.linalg.norm(rotation_part, axis=0).mean()  # Average scale factor
        }

    def transform_arkit_to_scenefun3d(self, points: np.ndarray) -> np.ndarray:
        """Transform ARKitScenes coordinates to SceneFun3D coordinates.

        This is the KEY INVERSION - instead of transforming affordance points to ARKit space,
        we transform ARKit object coordinates to SceneFun3D space.

        Args:
            points: Nx3 array of ARKitScenes coordinates

        Returns:
            Nx3 array of points in SceneFun3D coordinate system
        """
        if points.size == 0:
            return points

        # Convert to homogeneous coordinates
        if points.ndim == 1:
            points = points.reshape(1, -1)

        num_points = points.shape[0]
        homogeneous_points = np.hstack([points, np.ones((num_points, 1))])

        # Apply INVERSE transformation (ARKitScenes → SceneFun3D)
        transformed_homogeneous = homogeneous_points @ self.inverse_transform.T

        # Convert back to 3D coordinates
        transformed_points = transformed_homogeneous[:, :3]

        return transformed_points

    def transform_scenefun3d_to_arkit(self, points: np.ndarray) -> np.ndarray:
        """Transform SceneFun3D coordinates to ARKitScenes coordinates.

        This is the original transformation direction (kept for compatibility).

        Args:
            points: Nx3 array of SceneFun3D coordinates

        Returns:
            Nx3 array of points in ARKitScenes coordinate system
        """
        if points.size == 0:
            return points

        # Convert to homogeneous coordinates
        if points.ndim == 1:
            points = points.reshape(1, -1)

        num_points = points.shape[0]
        homogeneous_points = np.hstack([points, np.ones((num_points, 1))])

        # Apply forward transformation (SceneFun3D → ARKitScenes)
        transformed_homogeneous = homogeneous_points @ self.transform_matrix.T

        # Convert back to 3D coordinates
        transformed_points = transformed_homogeneous[:, :3]

        return transformed_points

    def transform_oriented_bounding_box(self, obb_center: np.ndarray,
                                      obb_size: np.ndarray,
                                      obb_rotation: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform an oriented bounding box from ARKitScenes to SceneFun3D coordinates.

        Args:
            obb_center: 3D center point in ARKitScenes coordinates
            obb_size: 3D size [width, height, depth]
            obb_rotation: 3x3 rotation matrix or 9-element flattened rotation

        Returns:
            Tuple of (transformed_center, transformed_size, transformed_rotation)
        """
        # Transform center point
        transformed_center = self.transform_arkit_to_scenefun3d(obb_center.reshape(1, -1)).flatten()

        # Transform rotation matrix
        if obb_rotation.shape == (9,):
            rotation_matrix = obb_rotation.reshape(3, 3)
        else:
            rotation_matrix = obb_rotation

        # Apply coordinate transformation to rotation
        # R_new = T_inv[:3,:3] @ R_old @ T[:3,:3]
        transform_rotation = self.inverse_transform[:3, :3]
        original_transform_rotation = self.transform_matrix[:3, :3]

        transformed_rotation = transform_rotation @ rotation_matrix @ original_transform_rotation

        # Size typically doesn't change with rigid transformation
        transformed_size = obb_size

        return transformed_center, transformed_size, transformed_rotation

    def get_bounding_box_corners(self, center: np.ndarray, size: np.ndarray,
                               rotation: np.ndarray) -> np.ndarray:
        """Get the 8 corners of an oriented bounding box.

        Args:
            center: 3D center point
            size: 3D size [width, height, depth]
            rotation: 3x3 rotation matrix

        Returns:
            8x3 array of corner coordinates
        """
        # Half extents
        half_extents = size / 2

        # Local corners (before rotation)
        local_corners = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ]) * half_extents

        # Apply rotation
        if rotation.shape == (9,):
            rotation = rotation.reshape(3, 3)
        rotated_corners = local_corners @ rotation.T

        # Translate to world position
        world_corners = rotated_corners + center

        return world_corners

    def get_axis_aligned_bbox_from_obb(self, center: np.ndarray, size: np.ndarray,
                                     rotation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box from oriented bounding box.

        Args:
            center: 3D center point
            size: 3D size [width, height, depth]
            rotation: 3x3 rotation matrix

        Returns:
            Tuple of (min_bounds, max_bounds) for axis-aligned box
        """
        corners = self.get_bounding_box_corners(center, size, rotation)
        min_bounds = np.min(corners, axis=0)
        max_bounds = np.max(corners, axis=0)

        return min_bounds, max_bounds

def test_coordinate_inversion():
    """Test the coordinate system inversion."""
    print("🧪 Testing Coordinate System Inversion")
    print("=" * 50)

    # Initialize transformer
    data_root = "/nas/jiachen/SceneFun3D/alignment/data_examples/scenefun3d"
    transformer = EnhancedCoordinateTransformer(data_root, "422203", "42445781")

    # Test with sample ARKitScenes data
    # ARKit toilet center from our previous analysis: [29.01, 249.20, -9.67] mm
    arkit_toilet_center = np.array([29.01, 249.20, -9.67])

    print(f"\n📊 ARKit toilet center: {arkit_toilet_center}")

    # Transform to SceneFun3D coordinates
    scenefun3d_toilet_center = transformer.transform_arkit_to_scenefun3d(arkit_toilet_center.reshape(1, -1))
    print(f"📊 SceneFun3D toilet center: {scenefun3d_toilet_center.flatten()}")

    # Test round-trip transformation
    back_to_arkit = transformer.transform_scenefun3d_to_arkit(scenefun3d_toilet_center)
    print(f"📊 Round-trip back to ARKit: {back_to_arkit.flatten()}")
    print(f"📊 Round-trip error: {np.linalg.norm(arkit_toilet_center - back_to_arkit.flatten()):.6f}")

    # Test bounding box transformation
    arkit_size = np.array([41.99, 79.52, 67.17])  # Toilet size
    arkit_rotation = np.array([
        0.4469533791606042, 0.0, 0.8945572518608953,
        0.0, 1.0, 0.0,
        -0.8945572518608953, 0.0, 0.4469533791606042
    ])

    transformed_center, transformed_size, transformed_rotation = transformer.transform_oriented_bounding_box(
        arkit_toilet_center, arkit_size, arkit_rotation
    )

    print(f"\n📦 Transformed bounding box:")
    print(f"   Center: {transformed_center}")
    print(f"   Size: {transformed_size}")
    print(f"   Rotation shape: {transformed_rotation.shape}")

    # Get axis-aligned bounds in SceneFun3D space
    min_bounds, max_bounds = transformer.get_axis_aligned_bbox_from_obb(
        transformed_center, transformed_size, transformed_rotation
    )

    print(f"\n📏 Axis-aligned bounds in SceneFun3D space:")
    print(f"   Min: {min_bounds}")
    print(f"   Max: {max_bounds}")
    print(f"   Size: {max_bounds - min_bounds}")


if __name__ == "__main__":
    test_coordinate_inversion()