"""
Point Cloud Utilities

Utilities for loading and processing point clouds from SceneFun3D laser scans.
"""

import numpy as np
import open3d as o3d
from typing import List, Tuple, Optional, Union


def load_laser_scan(ply_file: str) -> Tuple[np.ndarray, o3d.geometry.PointCloud]:
    """Load laser scan from PLY file.

    Args:
        ply_file: Path to PLY file

    Returns:
        Tuple of (points array, Open3D point cloud)
    """
    # Load with Open3D
    point_cloud = o3d.io.read_point_cloud(ply_file)

    if len(point_cloud.points) == 0:
        raise ValueError(f"No points loaded from {ply_file}")

    # Convert to numpy array
    points = np.asarray(point_cloud.points)

    print(f"Loaded laser scan: {len(points)} points from {ply_file}")

    return points, point_cloud


def extract_points_by_indices(points: np.ndarray, indices: List[int]) -> np.ndarray:
    """Extract subset of points by indices.

    Args:
        points: Nx3 array of all points
        indices: List of point indices to extract

    Returns:
        Mx3 array of selected points
    """
    if not indices:
        return np.empty((0, 3))

    # Validate indices
    max_index = len(points) - 1
    valid_indices = [i for i in indices if 0 <= i <= max_index]

    if len(valid_indices) != len(indices):
        print(f"Warning: {len(indices) - len(valid_indices)} indices out of bounds")

    return points[valid_indices]


def compute_point_cloud_bounds(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bounding box of point cloud.

    Args:
        points: Nx3 array of points

    Returns:
        Tuple of (min_bounds, max_bounds)
    """
    if points.size == 0:
        return np.zeros(3), np.zeros(3)

    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)

    return min_bounds, max_bounds


def compute_point_cloud_centroid(points: np.ndarray) -> np.ndarray:
    """Compute centroid of point cloud.

    Args:
        points: Nx3 array of points

    Returns:
        3D centroid coordinates
    """
    if points.size == 0:
        return np.zeros(3)

    return np.mean(points, axis=0)


def create_bounding_box_from_points(points: np.ndarray) -> dict:
    """Create axis-aligned bounding box from points.

    Args:
        points: Nx3 array of points

    Returns:
        Dictionary with bounding box information
    """
    if points.size == 0:
        return {
            'center': np.zeros(3),
            'size': np.zeros(3),
            'min_bounds': np.zeros(3),
            'max_bounds': np.zeros(3),
            'volume': 0.0
        }

    min_bounds, max_bounds = compute_point_cloud_bounds(points)
    size = max_bounds - min_bounds
    center = (min_bounds + max_bounds) / 2
    volume = np.prod(size)

    return {
        'center': center,
        'size': size,
        'min_bounds': min_bounds,
        'max_bounds': max_bounds,
        'volume': volume
    }


def filter_points_by_bounding_box(points: np.ndarray, min_bounds: np.ndarray,
                                  max_bounds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Filter points within a bounding box.

    Args:
        points: Nx3 array of points
        min_bounds: 3D minimum bounds
        max_bounds: 3D maximum bounds

    Returns:
        Tuple of (filtered_points, indices_of_filtered_points)
    """
    # Check which points are within bounds
    within_bounds = np.all((points >= min_bounds) & (points <= max_bounds), axis=1)

    filtered_points = points[within_bounds]
    filtered_indices = np.where(within_bounds)[0]

    return filtered_points, filtered_indices


def find_points_near_center(points: np.ndarray, center: np.ndarray,
                            radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """Find points within a spherical radius of a center point.

    Args:
        points: Nx3 array of points
        center: 3D center point
        radius: Search radius

    Returns:
        Tuple of (nearby_points, indices_of_nearby_points)
    """
    # Compute distances to center
    distances = np.linalg.norm(points - center, axis=1)

    # Find points within radius
    within_radius = distances <= radius

    nearby_points = points[within_radius]
    nearby_indices = np.where(within_radius)[0]

    return nearby_points, nearby_indices


def downsample_point_cloud(points: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample point cloud using voxel grid.

    Args:
        points: Nx3 array of points
        voxel_size: Size of voxel grid

    Returns:
        Tuple of (downsampled_points, indices_of_downsampled_points)
    """
    # Create Open3D point cloud
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)

    # Downsample
    downsampled_pc = pc.voxel_down_sample(voxel_size)

    # Get downsampled points
    downsampled_points = np.asarray(downsampled_pc.points)

    # Find indices of downsampled points in original cloud
    # (This is approximate due to voxel grid sampling)
    indices = []
    for down_point in downsampled_points:
        distances = np.linalg.norm(points - down_point, axis=1)
        closest_idx = np.argmin(distances)
        indices.append(closest_idx)

    return downsampled_points, np.array(indices)


def estimate_point_normals(points: np.ndarray, search_radius: float = None,
                           max_nn: int = 30) -> np.ndarray:
    """Estimate point normals using local neighborhood.

    Args:
        points: Nx3 array of points
        search_radius: Search radius for neighbors
        max_nn: Maximum number of nearest neighbors

    Returns:
        Nx3 array of normal vectors
    """
    # Create Open3D point cloud
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)

    # Estimate normals
    if search_radius is not None:
        pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(search_radius))
    else:
        pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(max_nn))

    # Return normals as numpy array
    return np.asarray(pc.normals)


def compute_point_cloud_statistics(points: np.ndarray) -> dict:
    """Compute comprehensive statistics for a point cloud.

    Args:
        points: Nx3 array of points

    Returns:
        Dictionary with statistics
    """
    if points.size == 0:
        return {'num_points': 0}

    min_bounds, max_bounds = compute_point_cloud_bounds(points)
    centroid = compute_point_cloud_centroid(points)
    size = max_bounds - min_bounds

    # Compute distances from centroid
    distances = np.linalg.norm(points - centroid, axis=1)

    # Compute density (approximate)
    volume = np.prod(size)
    density = len(points) / volume if volume > 0 else 0

    return {
        'num_points': len(points),
        'centroid': centroid,
        'min_bounds': min_bounds,
        'max_bounds': max_bounds,
        'size': size,
        'volume': volume,
        'density': density,
        'mean_distance_from_centroid': np.mean(distances),
        'std_distance_from_centroid': np.std(distances),
        'max_distance_from_centroid': np.max(distances)
    }


class PointCloudProcessor:
    """Helper class for processing point clouds."""

    def __init__(self, laser_scan_file: str):
        """Initialize with laser scan file."""
        self.laser_scan_file = laser_scan_file
        self.points, self.point_cloud = load_laser_scan(laser_scan_file)

    def get_points_by_indices(self, indices: List[int]) -> np.ndarray:
        """Get points by indices."""
        return extract_points_by_indices(self.points, indices)

    def get_annotation_points(self, annotation) -> np.ndarray:
        """Get points for a SceneFun3D annotation."""
        return self.get_points_by_indices(annotation.indices)

    def get_annotation_bbox(self, annotation) -> dict:
        """Get bounding box for annotation points."""
        points = self.get_annotation_points(annotation)
        return create_bounding_box_from_points(points)

    def visualize_points(self, indices: List[int] = None, color: List[float] = [1, 0, 0]):
        """Visualize points (optionally highlighting specific indices)."""
        vis_pc = o3d.geometry.PointCloud(self.point_cloud)

        if indices:
            # Color all points gray
            colors = np.array([[0.7, 0.7, 0.7]] * len(self.points))
            # Highlight selected points
            for idx in indices:
                if 0 <= idx < len(colors):
                    colors[idx] = color
            vis_pc.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([vis_pc])


def main():
    """Example usage of point cloud utilities."""
    # Load laser scan
    laser_scan_file = "/home/jiachen/scratch/SceneFun3D/alignment/data_examples/scenefun3d/visit_422203/422203_laser_scan.ply"

    processor = PointCloudProcessor(laser_scan_file)

    # Print statistics
    stats = compute_point_cloud_statistics(processor.points)
    print("Laser Scan Statistics:")
    print(f"  Points: {stats['num_points']:,}")
    print(f"  Bounds: {stats['min_bounds']} to {stats['max_bounds']}")
    print(f"  Size: {stats['size']}")
    print(f"  Centroid: {stats['centroid']}")
    print(f"  Density: {stats['density']:.2f} points/unit³")

    # Example: Extract first 1000 points
    sample_indices = list(range(1000))
    sample_points = processor.get_points_by_indices(sample_indices)
    sample_bbox = create_bounding_box_from_points(sample_points)

    print(f"\nSample (first 1000 points):")
    print(f"  Center: {sample_bbox['center']}")
    print(f"  Size: {sample_bbox['size']}")
    print(f"  Volume: {sample_bbox['volume']:.2f}")


if __name__ == "__main__":
    main()