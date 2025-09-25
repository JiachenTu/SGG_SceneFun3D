"""
Coordinate Transformation Utilities

Handles coordinate system alignment between ARKitScenes and SceneFun3D data.
"""

import numpy as np
import open3d as o3d
from typing import Tuple, Optional, Union


class CoordinateTransformer:
    """Handles coordinate transformations between ARKitScenes and SceneFun3D."""

    def __init__(self, transform_file: Optional[str] = None):
        """Initialize transformer with optional transform matrix file."""
        self.transform_matrix = np.eye(4)
        self.inverse_transform = np.eye(4)

        if transform_file:
            self.load_transform(transform_file)

    def load_transform(self, transform_file: str) -> None:
        """Load transformation matrix from file."""
        self.transform_matrix = np.load(transform_file)

        # Ensure it's a 4x4 matrix
        if self.transform_matrix.shape == (3, 4):
            # Convert 3x4 to 4x4
            bottom_row = np.array([[0, 0, 0, 1]])
            self.transform_matrix = np.vstack([self.transform_matrix, bottom_row])
        elif self.transform_matrix.shape != (4, 4):
            raise ValueError(f"Invalid transform matrix shape: {self.transform_matrix.shape}")

        # Compute inverse
        self.inverse_transform = np.linalg.inv(self.transform_matrix)

    def set_transform_matrix(self, matrix: np.ndarray) -> None:
        """Set transformation matrix directly."""
        if matrix.shape != (4, 4):
            raise ValueError(f"Transform matrix must be 4x4, got {matrix.shape}")

        self.transform_matrix = matrix.copy()
        self.inverse_transform = np.linalg.inv(self.transform_matrix)

    def transform_points(self, points: np.ndarray, inverse: bool = False) -> np.ndarray:
        """Transform 3D points using the transformation matrix.

        Args:
            points: Nx3 array of 3D points
            inverse: If True, apply inverse transformation

        Returns:
            Transformed Nx3 array of points
        """
        if points.size == 0:
            return points

        # Convert to homogeneous coordinates
        if points.ndim == 1:
            points = points.reshape(1, -1)

        num_points = points.shape[0]
        homogeneous_points = np.hstack([points, np.ones((num_points, 1))])

        # Apply transformation
        transform = self.inverse_transform if inverse else self.transform_matrix
        transformed_homogeneous = homogeneous_points @ transform.T

        # Convert back to 3D coordinates
        transformed_points = transformed_homogeneous[:, :3]

        return transformed_points

    def transform_point_cloud(self, point_cloud: o3d.geometry.PointCloud,
                              inverse: bool = False) -> o3d.geometry.PointCloud:
        """Transform an Open3D point cloud."""
        points = np.asarray(point_cloud.points)
        transformed_points = self.transform_points(points, inverse=inverse)

        # Create new point cloud
        transformed_pc = o3d.geometry.PointCloud()
        transformed_pc.points = o3d.utility.Vector3dVector(transformed_points)

        # Transform normals if available
        if point_cloud.has_normals():
            normals = np.asarray(point_cloud.normals)
            # For normals, only apply rotation part
            rotation_matrix = self.transform_matrix[:3, :3]
            if inverse:
                rotation_matrix = self.inverse_transform[:3, :3]
            transformed_normals = normals @ rotation_matrix.T
            transformed_pc.normals = o3d.utility.Vector3dVector(transformed_normals)

        # Copy colors if available
        if point_cloud.has_colors():
            transformed_pc.colors = point_cloud.colors

        return transformed_pc

    def align_point_indices_to_3d(self, indices: list, laser_scan_points: np.ndarray,
                                  target_coordinate_system: str = "arkitscenes") -> np.ndarray:
        """Convert point indices to 3D coordinates in target coordinate system.

        Args:
            indices: List of point indices from SceneFun3D
            laser_scan_points: Nx3 array of all laser scan points
            target_coordinate_system: "arkitscenes" or "scenefun3d"

        Returns:
            Mx3 array of 3D coordinates in target system
        """
        if not indices:
            return np.empty((0, 3))

        # Get selected points
        selected_points = laser_scan_points[indices]

        # Transform to target coordinate system
        if target_coordinate_system.lower() == "arkitscenes":
            # Transform from SceneFun3D to ARKitScenes coordinates
            return self.transform_points(selected_points, inverse=False)
        elif target_coordinate_system.lower() == "scenefun3d":
            # Keep in SceneFun3D coordinates
            return selected_points
        else:
            raise ValueError(f"Unknown coordinate system: {target_coordinate_system}")

    def get_transformation_info(self) -> dict:
        """Get information about the transformation."""
        # Decompose transformation matrix
        rotation = self.transform_matrix[:3, :3]
        translation = self.transform_matrix[:3, 3]

        # Compute rotation angle and axis
        trace = np.trace(rotation)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))

        # Compute scale (assuming uniform scaling)
        scale = np.linalg.norm(rotation, axis=0).mean()

        return {
            'matrix': self.transform_matrix,
            'rotation_matrix': rotation,
            'translation': translation,
            'rotation_angle_deg': np.degrees(angle),
            'scale': scale,
            'determinant': np.linalg.det(self.transform_matrix)
        }

    def compute_alignment_error(self, points_a: np.ndarray, points_b: np.ndarray) -> dict:
        """Compute alignment error between two sets of corresponding points.

        Args:
            points_a: Nx3 source points
            points_b: Nx3 target points

        Returns:
            Dictionary with error statistics
        """
        if points_a.shape != points_b.shape:
            raise ValueError("Point sets must have same shape")

        # Transform points_a to align with points_b
        transformed_a = self.transform_points(points_a)

        # Compute errors
        errors = np.linalg.norm(transformed_a - points_b, axis=1)

        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'num_points': len(errors)
        }


def estimate_transform_from_correspondences(source_points: np.ndarray,
                                            target_points: np.ndarray) -> np.ndarray:
    """Estimate transformation matrix from point correspondences using SVD.

    Args:
        source_points: Nx3 source points
        target_points: Nx3 target points

    Returns:
        4x4 transformation matrix
    """
    if source_points.shape != target_points.shape:
        raise ValueError("Source and target points must have same shape")

    if source_points.shape[1] != 3:
        raise ValueError("Points must be 3D (Nx3)")

    # Center the points
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)

    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid

    # Compute cross-covariance matrix
    H = source_centered.T @ target_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation
    R = Vt.T @ U.T

    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = target_centroid - R @ source_centroid

    # Construct 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t

    return transform


def main():
    """Example usage of coordinate transformation utilities."""
    # Load transform file
    transform_file = "/home/jiachen/scratch/SceneFun3D/alignment/data_examples/scenefun3d/visit_422203/42445781/42445781_transform.npy"

    transformer = CoordinateTransformer(transform_file)

    # Print transformation info
    info = transformer.get_transformation_info()
    print("Transformation Matrix:")
    print(info['matrix'])
    print(f"\nRotation angle: {info['rotation_angle_deg']:.2f} degrees")
    print(f"Translation: {info['translation']}")
    print(f"Scale: {info['scale']:.3f}")

    # Example point transformation
    test_points = np.array([[0, 0, 0], [100, 100, 100], [50, 200, 150]])
    transformed = transformer.transform_points(test_points)

    print(f"\nExample point transformation:")
    for i, (orig, trans) in enumerate(zip(test_points, transformed)):
        print(f"Point {i}: {orig} → {trans}")


if __name__ == "__main__":
    main()