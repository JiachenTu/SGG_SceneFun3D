"""
Spatial Analysis for Object-Affordance Mapping

Analyzes spatial relationships between ARKitScenes objects and SceneFun3D affordances.
"""

import sys
import os

# Handle imports for both standalone and integrated execution
try:
    # Try relative imports first (when run as module)
    from utils.arkitscenes_parser import ARKitScenesParser, ARKitObject
    from utils.scenefun3d_parser import SceneFun3DParser, TaskDescription, Annotation
    from utils.coordinate_transform import CoordinateTransformer
    from utils.point_cloud_utils import PointCloudProcessor
except ImportError:
    # Fall back to path-based imports (when run standalone)
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
    from arkitscenes_parser import ARKitScenesParser, ARKitObject
    from scenefun3d_parser import SceneFun3DParser, TaskDescription, Annotation
    from coordinate_transform import CoordinateTransformer
    from point_cloud_utils import PointCloudProcessor

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass



@dataclass
class SpatialRelationship:
    """Represents spatial relationship between object and affordance."""
    object_id: str
    object_label: str
    annotation_id: str
    task_description: str
    overlap_ratio: float
    distance: float
    confidence: float
    affordance_type: str
    motion_type: Optional[str] = None


class SpatialAnalyzer:
    """Analyzes spatial relationships between objects and affordances."""

    def __init__(self, arkitscenes_parser: ARKitScenesParser,
                 scenefun3d_parser: SceneFun3DParser,
                 transformer: CoordinateTransformer,
                 point_processor: PointCloudProcessor):
        """Initialize spatial analyzer."""
        self.arkitscenes_parser = arkitscenes_parser
        self.scenefun3d_parser = scenefun3d_parser
        self.transformer = transformer
        self.point_processor = point_processor

    def compute_object_affordance_overlap(self, obj: ARKitObject,
                                          annotation: Annotation) -> Tuple[float, float]:
        """Compute spatial overlap between object and affordance.

        Args:
            obj: ARKitScenes object
            annotation: SceneFun3D annotation

        Returns:
            Tuple of (overlap_ratio, distance_between_centers)
        """
        # Get affordance points in ARKitScenes coordinate system
        affordance_points = self.point_processor.get_annotation_points(annotation)
        transformed_points = self.transformer.transform_points(affordance_points)

        if len(transformed_points) == 0:
            return 0.0, float('inf')

        # Get object bounding box corners
        object_corners = obj.obb.get_corners()
        object_min = np.min(object_corners, axis=0)
        object_max = np.max(object_corners, axis=0)

        # Check how many affordance points are inside object bbox
        inside_bbox = np.all((transformed_points >= object_min) &
                             (transformed_points <= object_max), axis=1)
        overlap_ratio = np.sum(inside_bbox) / len(transformed_points)

        # Compute distance between centers
        affordance_center = np.mean(transformed_points, axis=0)
        object_center = obj.obb.centroid
        distance = np.linalg.norm(affordance_center - object_center)

        return overlap_ratio, distance

    def find_parent_object_for_affordance(self, annotation: Annotation,
                                          overlap_threshold: float = 0.1,
                                          distance_threshold: float = 500.0) -> Optional[ARKitObject]:
        """Find the most likely parent object for an affordance.

        Args:
            annotation: SceneFun3D annotation
            overlap_threshold: Minimum overlap ratio to consider
            distance_threshold: Maximum distance in mm to consider

        Returns:
            Best matching ARKitScenes object or None
        """
        best_object = None
        best_score = -1

        for obj in self.arkitscenes_parser.get_objects():
            overlap_ratio, distance = self.compute_object_affordance_overlap(obj, annotation)

            # Skip if criteria not met
            if overlap_ratio < overlap_threshold and distance > distance_threshold:
                continue

            # Compute combined score (higher is better)
            # Favor higher overlap and lower distance
            score = overlap_ratio * 0.7 + (1.0 / (1.0 + distance / 100.0)) * 0.3

            if score > best_score:
                best_score = score
                best_object = obj

        return best_object

    def analyze_task_spatial_relationships(self, task: TaskDescription) -> List[SpatialRelationship]:
        """Analyze spatial relationships for a specific task.

        Args:
            task: TaskDescription to analyze

        Returns:
            List of spatial relationships found
        """
        relationships = []

        for annotation in task.annotations:
            # Find parent object
            parent_object = self.find_parent_object_for_affordance(annotation)

            if parent_object:
                overlap_ratio, distance = self.compute_object_affordance_overlap(
                    parent_object, annotation)

                # Compute confidence based on overlap and distance
                confidence = min(1.0, overlap_ratio + (1.0 / (1.0 + distance / 100.0)) * 0.5)

                # Get motion type if available
                motion_type = None
                if annotation.motion:
                    motion_type = annotation.motion.motion_type

                relationship = SpatialRelationship(
                    object_id=parent_object.uid,
                    object_label=parent_object.label,
                    annotation_id=annotation.annot_id,
                    task_description=task.description,
                    overlap_ratio=overlap_ratio,
                    distance=distance,
                    confidence=confidence,
                    affordance_type=task.infer_affordance_type(),
                    motion_type=motion_type
                )

                relationships.append(relationship)

        return relationships

    def analyze_all_spatial_relationships(self) -> List[SpatialRelationship]:
        """Analyze spatial relationships for all tasks.

        Returns:
            List of all spatial relationships found
        """
        all_relationships = []

        for task in self.scenefun3d_parser.get_task_descriptions():
            task_relationships = self.analyze_task_spatial_relationships(task)
            all_relationships.extend(task_relationships)

        return all_relationships

    def group_affordances_by_object(self, relationships: List[SpatialRelationship]) -> Dict[str, List[SpatialRelationship]]:
        """Group affordances by their parent objects.

        Args:
            relationships: List of spatial relationships

        Returns:
            Dictionary mapping object_id to list of relationships
        """
        grouped = {}

        for rel in relationships:
            if rel.object_id not in grouped:
                grouped[rel.object_id] = []
            grouped[rel.object_id].append(rel)

        return grouped

    def find_objects_with_affordance_type(self, affordance_type: str,
                                          relationships: List[SpatialRelationship]) -> List[str]:
        """Find objects that have a specific affordance type.

        Args:
            affordance_type: Type of affordance to search for
            relationships: List of spatial relationships

        Returns:
            List of object IDs with the specified affordance
        """
        object_ids = []

        for rel in relationships:
            if rel.affordance_type.lower() == affordance_type.lower():
                if rel.object_id not in object_ids:
                    object_ids.append(rel.object_id)

        return object_ids

    def compute_spatial_statistics(self, relationships: List[SpatialRelationship]) -> Dict:
        """Compute statistics about spatial relationships.

        Args:
            relationships: List of spatial relationships

        Returns:
            Dictionary with statistics
        """
        if not relationships:
            return {}

        overlaps = [rel.overlap_ratio for rel in relationships]
        distances = [rel.distance for rel in relationships]
        confidences = [rel.confidence for rel in relationships]

        # Group by affordance type
        affordance_types = {}
        for rel in relationships:
            aff_type = rel.affordance_type
            if aff_type not in affordance_types:
                affordance_types[aff_type] = []
            affordance_types[aff_type].append(rel)

        # Group by object label
        object_labels = {}
        for rel in relationships:
            label = rel.object_label
            if label not in object_labels:
                object_labels[label] = []
            object_labels[label].append(rel)

        return {
            'total_relationships': len(relationships),
            'mean_overlap': np.mean(overlaps),
            'std_overlap': np.std(overlaps),
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'mean_confidence': np.mean(confidences),
            'high_confidence_count': sum(1 for c in confidences if c > 0.7),
            'affordance_type_counts': {k: len(v) for k, v in affordance_types.items()},
            'object_label_counts': {k: len(v) for k, v in object_labels.items()}
        }

    def print_spatial_analysis_summary(self, relationships: List[SpatialRelationship]) -> None:
        """Print summary of spatial analysis."""
        print("=" * 60)
        print("SPATIAL ANALYSIS SUMMARY")
        print("=" * 60)

        stats = self.compute_spatial_statistics(relationships)

        print(f"Total relationships found: {stats.get('total_relationships', 0)}")
        print(f"Mean overlap ratio: {stats.get('mean_overlap', 0):.3f}")
        print(f"Mean distance: {stats.get('mean_distance', 0):.1f} mm")
        print(f"Mean confidence: {stats.get('mean_confidence', 0):.3f}")
        print(f"High confidence (>0.7): {stats.get('high_confidence_count', 0)}")

        print(f"\nAffordance types found:")
        for aff_type, count in stats.get('affordance_type_counts', {}).items():
            print(f"  {aff_type}: {count}")

        print(f"\nObject labels with affordances:")
        for label, count in stats.get('object_label_counts', {}).items():
            print(f"  {label}: {count}")

        print(f"\nDetailed relationships:")
        for rel in relationships:
            print(f"  {rel.object_label} → {rel.affordance_type}")
            print(f"    Task: \"{rel.task_description}\"")
            print(f"    Overlap: {rel.overlap_ratio:.3f}, Distance: {rel.distance:.1f}mm")
            print(f"    Confidence: {rel.confidence:.3f}")
            if rel.motion_type:
                print(f"    Motion: {rel.motion_type}")
            print()


def main():
    """Example usage of spatial analysis."""
    # File paths
    arkitscenes_file = "/home/jiachen/scratch/SceneFun3D/alignment/data_examples/arkitscenes/video_42445781/42445781_3dod_annotation.json"
    scenefun3d_dir = "/home/jiachen/scratch/SceneFun3D/alignment/data_examples/scenefun3d/visit_422203"
    transform_file = "/home/jiachen/scratch/SceneFun3D/alignment/data_examples/scenefun3d/visit_422203/42445781/42445781_transform.npy"
    laser_scan_file = "/home/jiachen/scratch/SceneFun3D/alignment/data_examples/scenefun3d/visit_422203/422203_laser_scan.ply"

    print("Loading data...")

    # Initialize parsers
    arkitscenes_parser = ARKitScenesParser(arkitscenes_file)
    arkitscenes_parser.load()

    scenefun3d_parser = SceneFun3DParser(scenefun3d_dir)
    scenefun3d_parser.load("422203")

    # Initialize transformer and point processor
    transformer = CoordinateTransformer(transform_file)
    point_processor = PointCloudProcessor(laser_scan_file)

    print("Analyzing spatial relationships...")

    # Create spatial analyzer
    analyzer = SpatialAnalyzer(arkitscenes_parser, scenefun3d_parser,
                               transformer, point_processor)

    # Analyze all relationships
    relationships = analyzer.analyze_all_spatial_relationships()

    # Print summary
    analyzer.print_spatial_analysis_summary(relationships)

    # Example: Focus on toilet-related tasks
    print("\n" + "=" * 60)
    print("TOILET-SPECIFIC ANALYSIS")
    print("=" * 60)

    toilet_relationships = [rel for rel in relationships if 'toilet' in rel.object_label.lower()]
    for rel in toilet_relationships:
        print(f"Task: \"{rel.task_description}\"")
        print(f"Affordance: {rel.affordance_type} ({rel.motion_type})")
        print(f"Confidence: {rel.confidence:.3f}")
        print()


if __name__ == "__main__":
    main()