"""
Enhanced Coordinate Transformation Utilities - V2.0

This module implements coordinate system inversion - transforming ARKitScenes data
to SceneFun3D coordinate system instead of the reverse.

Key Features:
- Transform ARKit object bounding boxes, centroids, vertices to laser scan coordinates
- Keep affordance points in native SceneFun3D coordinate system (no transformation)
- Use inverse transformation matrix for ARKit → SceneFun3D conversion
- Batch processing support for efficiency
- Integration with UnifiedDataLoader
- Comprehensive validation and error handling
"""

import numpy as np
import open3d as o3d
import os
from typing import Tuple, Optional, Union, List, Dict, Any
from pathlib import Path
import sys
import logging
from dataclasses import dataclass


@dataclass
class TransformationInfo:
    """Container for transformation information."""
    transform_matrix: np.ndarray
    inverse_transform: np.ndarray
    rotation_angle_deg: float
    translation: np.ndarray
    scale: float
    determinant: float
    is_valid: bool = True
    validation_errors: List[str] = None

    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []


class EnhancedCoordinateTransformer:
    """
    Enhanced coordinate transformer with coordinate system inversion support.

    Key Features:
    - Works with any data source (UnifiedDataLoader or direct matrix)
    - Batch processing for efficiency
    - Comprehensive validation
    - Memory-efficient operations
    """

    def __init__(self, transform_matrix: Optional[np.ndarray] = None,
                 data_root: Optional[str] = None, visit_id: Optional[str] = None,
                 video_id: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize coordinate transformer.

        Args:
            transform_matrix: Direct transformation matrix (4x4)
            data_root: Data root for SceneFun3D DataParser (alternative)
            visit_id: Visit ID for loading transform matrix
            video_id: Video ID for loading transform matrix
            logger: Optional logger
        """
        self.logger = logger or logging.getLogger(__name__)

        # Load or use provided transformation matrix
        if transform_matrix is not None:
            self.transform_matrix = transform_matrix
            self.data_root = None
            self.visit_id = None
            self.video_id = None
        elif all([data_root, visit_id, video_id]):
            self.data_root = data_root
            self.visit_id = visit_id
            self.video_id = video_id
            self.transform_matrix = self._load_transform_matrix()
        else:
            raise ValueError("Either provide transform_matrix directly or data_root/visit_id/video_id")

        # Compute inverse transformation (ARKitScenes → SceneFun3D)
        try:
            self.inverse_transform = np.linalg.inv(self.transform_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("Transformation matrix is not invertible")

        # Cache transformation info
        self._transform_info = None

        self.logger.info(f"✅ Initialized coordinate transformer")
        self.logger.info(f"📊 Transform shape: {self.transform_matrix.shape}")
        self.logger.info(f"📊 Determinant: {np.linalg.det(self.transform_matrix):.6f}")

    def _load_transform_matrix(self) -> np.ndarray:
        """Load transformation matrix using SceneFun3D DataParser."""
        # Dynamic import to avoid circular dependencies
        # Use direct DataParser import with correct path
        scenefun3d_root = Path("/home/jiachen/scratch/SceneFun3D/scenefun3d")
        original_cwd = os.getcwd()
        os.chdir(str(scenefun3d_root))
        if str(scenefun3d_root) not in sys.path:
            sys.path.insert(0, str(scenefun3d_root))

        try:
            from utils.data_parser import DataParser
            parser = DataParser(self.data_root)
            return parser.get_transform(self.visit_id, self.video_id)
        finally:
            os.chdir(original_cwd)

    def get_transformation_info(self) -> TransformationInfo:
        """Get comprehensive transformation information."""
        if self._transform_info is not None:
            return self._transform_info

        try:
            rotation_part = self.transform_matrix[:3, :3]
            translation_part = self.transform_matrix[:3, 3]

            # Extract rotation angle (assuming rotation around Z-axis)
            rotation_angle_rad = np.arctan2(rotation_part[1, 0], rotation_part[0, 0])
            rotation_angle_deg = np.degrees(rotation_angle_rad)

            # Calculate determinant
            determinant = np.linalg.det(rotation_part)

            # Calculate scale
            scale = np.linalg.norm(rotation_part, axis=0).mean()

            # Validate transformation
            validation_errors = []
            is_valid = True

            if abs(determinant - 1.0) > 0.1:  # Allow small numerical errors
                validation_errors.append(f"Determinant {determinant:.3f} suggests non-rigid transformation")
                is_valid = False

            if abs(scale - 1.0) > 0.1:
                validation_errors.append(f"Scale {scale:.3f} suggests non-uniform scaling")

            self._transform_info = TransformationInfo(
                transform_matrix=self.transform_matrix,
                inverse_transform=self.inverse_transform,
                rotation_angle_deg=rotation_angle_deg,
                translation=translation_part,
                scale=scale,
                determinant=determinant,
                is_valid=is_valid,
                validation_errors=validation_errors
            )

            if validation_errors:
                self.logger.warning(f"Transformation validation warnings: {validation_errors}")

            return self._transform_info

        except Exception as e:
            self.logger.error(f"Failed to analyze transformation: {e}")
            return TransformationInfo(
                transform_matrix=self.transform_matrix,
                inverse_transform=self.inverse_transform,
                rotation_angle_deg=0.0,
                translation=np.zeros(3),
                scale=1.0,
                determinant=0.0,
                is_valid=False,
                validation_errors=[str(e)]
            )

    def transform_arkit_to_scenefun3d(self, points: np.ndarray) -> np.ndarray:
        """Transform ARKitScenes coordinates to SceneFun3D coordinates.

        This is the KEY INVERSION - instead of transforming affordance points to ARKit space,
        we transform ARKit object coordinates to SceneFun3D space.

        Args:
            points: Nx3 array of ARKitScenes coordinates or single 3D point

        Returns:
            Nx3 array of points in SceneFun3D coordinate system
        """
        return self._apply_transformation(points, self.inverse_transform)

    def transform_scenefun3d_to_arkit(self, points: np.ndarray) -> np.ndarray:
        """Transform SceneFun3D coordinates to ARKitScenes coordinates.

        This is the original transformation direction (kept for compatibility).

        Args:
            points: Nx3 array of SceneFun3D coordinates

        Returns:
            Nx3 array of points in ARKitScenes coordinate system
        """
        return self._apply_transformation(points, self.transform_matrix)

    def _apply_transformation(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """
        Apply homogeneous transformation to points.

        Args:
            points: Nx3 array of 3D points or single 3D point
            transform: 4x4 transformation matrix

        Returns:
            Nx3 array of transformed points
        """
        if points.size == 0:
            return points

        # Ensure points are in correct shape
        original_shape = points.shape
        if points.ndim == 1 and len(points) == 3:
            points = points.reshape(1, -1)
        elif points.ndim == 2 and points.shape[1] != 3:
            raise ValueError(f"Points must be Nx3 array, got shape {points.shape}")

        # Convert to homogeneous coordinates
        num_points = points.shape[0]
        homogeneous_points = np.hstack([points, np.ones((num_points, 1))])

        # Apply transformation
        transformed_homogeneous = homogeneous_points @ transform.T

        # Convert back to 3D coordinates
        transformed_points = transformed_homogeneous[:, :3]

        # Return in original shape if input was single point
        if original_shape == (3,):
            return transformed_points.flatten()

        return transformed_points

    def batch_transform_objects(self, objects_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch transform multiple objects from ARKit to SceneFun3D coordinates.

        Args:
            objects_data: List of object dictionaries with 'center', 'size', 'rotation'

        Returns:
            List of transformed object dictionaries
        """
        transformed_objects = []

        for obj_data in objects_data:
            try:
                # Extract object properties
                center = np.array(obj_data['center'])
                size = np.array(obj_data.get('size', [1.0, 1.0, 1.0]))
                rotation = np.array(obj_data.get('rotation', np.eye(3)))

                # Transform oriented bounding box
                transformed_center, transformed_size, transformed_rotation = \
                    self.transform_oriented_bounding_box(center, size, rotation)

                # Create transformed object data
                transformed_obj = obj_data.copy()
                transformed_obj.update({
                    'center': transformed_center.tolist(),
                    'size': transformed_size.tolist(),
                    'rotation': transformed_rotation.tolist(),
                    'original_center': center.tolist(),
                    'coordinate_system': 'SceneFun3D',
                })

                transformed_objects.append(transformed_obj)

            except Exception as e:
                self.logger.error(f"Failed to transform object {obj_data.get('id', 'unknown')}: {e}")
                # Keep original object with error flag
                error_obj = obj_data.copy()
                error_obj['transformation_error'] = str(e)
                transformed_objects.append(error_obj)

        return transformed_objects

    def validate_round_trip_accuracy(self, test_points: Optional[np.ndarray] = None,
                                   tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Validate transformation accuracy with round-trip test.

        Args:
            test_points: Optional test points, generates random if None
            tolerance: Acceptable error tolerance

        Returns:
            Dictionary with validation results
        """
        if test_points is None:
            # Generate test points in reasonable range (meters)
            np.random.seed(42)  # Reproducible results
            test_points = np.random.uniform(-10, 10, (10, 3))  # ±10m range

        # Perform round-trip transformation
        scenefun3d_points = self.transform_arkit_to_scenefun3d(test_points)
        back_to_arkit = self.transform_scenefun3d_to_arkit(scenefun3d_points)

        # Calculate errors
        errors = np.linalg.norm(test_points - back_to_arkit, axis=1)
        max_error = np.max(errors)
        mean_error = np.mean(errors)

        # Validation results (adjusted tolerance for meter-scale coordinates)
        # Use 1μm tolerance for meter coordinates
        meter_tolerance = max(tolerance, 1e-6)  # At least 1μm precision
        is_accurate = max_error < meter_tolerance

        return {
            'max_error': float(max_error),
            'mean_error': float(mean_error),
            'is_accurate': is_accurate,
            'tolerance': float(meter_tolerance),
            'test_points_count': len(test_points),
            'coordinate_units': 'meters',
            'all_errors': errors.tolist()
        }

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

def create_transformer_from_data(data_root: str, visit_id: str, video_id: str) -> EnhancedCoordinateTransformer:
    """
    Factory function to create transformer with any data root.

    Args:
        data_root: Root directory containing SceneFun3D data
        visit_id: Visit/scene identifier
        video_id: Video sequence identifier

    Returns:
        Initialized EnhancedCoordinateTransformer
    """
    return EnhancedCoordinateTransformer(
        data_root=data_root,
        visit_id=visit_id,
        video_id=video_id
    )


def create_transformer_from_matrix(transform_matrix: np.ndarray) -> EnhancedCoordinateTransformer:
    """
    Factory function to create transformer from transformation matrix.

    Args:
        transform_matrix: 4x4 transformation matrix

    Returns:
        Initialized EnhancedCoordinateTransformer
    """
    return EnhancedCoordinateTransformer(transform_matrix=transform_matrix)


def test_coordinate_inversion(data_root: str = None):
    """Test the coordinate system inversion with configurable data root."""
    print("🧪 Testing Enhanced Coordinate System Inversion - V2.0")
    print("=" * 60)

    # Use provided data root or default
    if data_root is None:
        data_root = "/nas/jiachen/SceneFun3D/alignment/data_examples/scenefun3d"

    print(f"📁 Using data root: {data_root}")

    try:
        # Initialize transformer
        transformer = create_transformer_from_data(data_root, "422203", "42445781")

        # Get transformation info
        info = transformer.get_transformation_info()
        print(f"\n📊 Transformation Info:")
        print(f"   Rotation angle: {info.rotation_angle_deg:.2f}°")
        print(f"   Translation: {info.translation}")
        print(f"   Scale: {info.scale:.3f}")
        print(f"   Determinant: {info.determinant:.6f}")
        print(f"   Valid: {info.is_valid}")
        if info.validation_errors:
            print(f"   Warnings: {info.validation_errors}")

        # Test with sample ARKitScenes data
        arkit_toilet_center = np.array([29.01, 249.20, -9.67])
        print(f"\n📊 ARKit toilet center: {arkit_toilet_center}")

        # Transform to SceneFun3D coordinates
        scenefun3d_toilet_center = transformer.transform_arkit_to_scenefun3d(arkit_toilet_center)
        print(f"📊 SceneFun3D toilet center: {scenefun3d_toilet_center}")

        # Test round-trip accuracy
        round_trip_results = transformer.validate_round_trip_accuracy(
            test_points=arkit_toilet_center.reshape(1, -1)
        )
        print(f"\n🔄 Round-trip validation:")
        print(f"   Max error: {round_trip_results['max_error']:.8f}")
        print(f"   Accurate: {round_trip_results['is_accurate']}")

        # Test bounding box transformation
        arkit_size = np.array([41.99, 79.52, 67.17])
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

        # Get axis-aligned bounds
        min_bounds, max_bounds = transformer.get_axis_aligned_bbox_from_obb(
            transformed_center, transformed_size, transformed_rotation
        )

        print(f"\n📏 Axis-aligned bounds in SceneFun3D space:")
        print(f"   Min: {min_bounds}")
        print(f"   Max: {max_bounds}")
        print(f"   Size: {max_bounds - min_bounds}")

        # Test batch transformation
        test_objects = [
            {
                'id': 'toilet',
                'center': [29.01, 249.20, -9.67],
                'size': [41.99, 79.52, 67.17],
                'rotation': arkit_rotation.reshape(3, 3)
            },
            {
                'id': 'sink',
                'center': [81.51, 280.84, 67.49],
                'size': [56.94, 23.10, 43.70],
                'rotation': np.eye(3)
            }
        ]

        transformed_objects = transformer.batch_transform_objects(test_objects)
        print(f"\n🔄 Batch transformation:")
        for obj in transformed_objects:
            print(f"   {obj['id']}: {obj['center'][:3]} -> {obj.get('transformation_error', 'OK')}")

        print("\n✅ Coordinate transformation test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test coordinate transformation")
    parser.add_argument("--data-root", help="SceneFun3D data root directory")
    args = parser.parse_args()

    test_coordinate_inversion(args.data_root)