"""
Enhanced Spatial Analysis with Coordinate System Inversion

This module implements the NEW spatial analysis approach:
1. Transform ARKitScenes objects TO SceneFun3D coordinate system
2. Keep affordance points in native SceneFun3D coordinates (no transformation)
3. Remove fallback logic - allow floating affordances if no confident match
4. Use mesh-based analysis for better accuracy

Key Changes from Original:
- Coordinate system inversion (ARKit → SceneFun3D instead of SceneFun3D → ARKit)
- Elimination of uncertain fallback connections
- Enhanced geometric analysis using mesh data
- Strict confidence thresholds
"""

import sys
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# Import enhanced coordinate transformer
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from enhanced_coordinate_transform import EnhancedCoordinateTransformer

# Import SceneFun3D toolkit
scenefun3d_root = Path(__file__).parent.parent.parent / "scenefun3d"
sys.path.insert(0, str(scenefun3d_root))
from utils.data_parser import DataParser

# Import original parsers for ARKitScenes data
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from arkitscenes_parser import ARKitScenesParser, ARKitObject


@dataclass
class EnhancedSpatialRelationship:
    """Enhanced spatial relationship with stricter confidence metrics."""
    object_id: str
    object_label: str
    annotation_id: str
    task_description: str
    overlap_ratio: float
    distance: float
    confidence: float
    affordance_type: str
    motion_type: Optional[str] = None
    mesh_proximity_score: float = 0.0
    geometric_plausibility: float = 0.0

    def is_confident_match(self, threshold: float = 0.3) -> bool:
        """Check if this is a confident spatial match."""
        return self.confidence >= threshold


class EnhancedSpatialAnalyzer:
    """Enhanced spatial analyzer with coordinate inversion and mesh analysis."""

    def __init__(self, data_root: str, visit_id: str, video_id: str):
        """Initialize with enhanced coordinate system."""
        self.data_root = data_root
        self.visit_id = visit_id
        self.video_id = video_id

        # Initialize enhanced coordinate transformer
        self.transformer = EnhancedCoordinateTransformer(data_root, visit_id, video_id)

        # Initialize SceneFun3D DataParser
        self.scenefun3d_parser = DataParser(data_root)

        # Initialize ARKitScenes parser
        arkitscenes_file = Path(__file__).parent.parent / f"data_examples/arkitscenes/video_{video_id}/{video_id}_3dod_annotation.json"
        self.arkitscenes_parser = ARKitScenesParser(str(arkitscenes_file))
        self.arkitscenes_parser.load()

        print(f"✅ Enhanced Spatial Analyzer initialized")
        print(f"   📊 ARKitScenes objects: {len(self.arkitscenes_parser.get_objects())}")
        print(f"   📊 Data root: {data_root}")

    def load_scenefun3d_data(self):
        """Load SceneFun3D annotations and task descriptions."""
        self.annotations = self.scenefun3d_parser.get_annotations(self.visit_id)
        self.descriptions = self.scenefun3d_parser.get_descriptions(self.visit_id)
        self.motions = self.scenefun3d_parser.get_motions(self.visit_id)

        # Create lookup dictionaries
        self.annotation_lookup = {ann['annot_id']: ann for ann in self.annotations}
        self.motion_lookup = {motion['annot_id']: motion for motion in self.motions}

        print(f"   📊 Loaded {len(self.annotations)} annotations, {len(self.descriptions)} descriptions, {len(self.motions)} motions")

    def get_affordance_points_in_scenefun3d(self, annotation_id: str) -> np.ndarray:
        """Get affordance points in SceneFun3D coordinate system (NO transformation needed).

        This is the KEY CHANGE - affordance points stay in their native coordinate system.
        """
        annotation = self.annotation_lookup.get(annotation_id)
        if not annotation:
            return np.empty((0, 3))

        # Load laser scan
        laser_scan = self.scenefun3d_parser.get_laser_scan(self.visit_id)
        laser_points = np.asarray(laser_scan.points)

        # Extract affordance points by indices - NO TRANSFORMATION
        indices = annotation['indices']
        if not indices:
            return np.empty((0, 3))

        # Validate indices
        max_index = len(laser_points) - 1
        valid_indices = [i for i in indices if 0 <= i <= max_index]

        if len(valid_indices) != len(indices):
            print(f"   ⚠️ Warning: {len(indices) - len(valid_indices)} indices out of bounds for annotation {annotation_id}")

        return laser_points[valid_indices]

    def transform_arkit_object_to_scenefun3d(self, obj: ARKitObject) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform ARKitScenes object to SceneFun3D coordinate system.

        This is the KEY INVERSION - we transform ARKit objects instead of affordance points.

        Returns:
            Tuple of (transformed_center, axis_aligned_min, axis_aligned_max)
        """
        # Transform object center, size, and rotation to SceneFun3D coordinates
        transformed_center, transformed_size, transformed_rotation = self.transformer.transform_oriented_bounding_box(
            obj.obb.centroid, obj.obb.axes_lengths, obj.obb.normalized_axes
        )

        # Get axis-aligned bounding box in SceneFun3D space
        min_bounds, max_bounds = self.transformer.get_axis_aligned_bbox_from_obb(
            transformed_center, transformed_size, transformed_rotation
        )

        return transformed_center, min_bounds, max_bounds

    def compute_enhanced_overlap(self, obj: ARKitObject, annotation_id: str) -> Tuple[float, float, float]:
        """Compute enhanced spatial overlap with mesh-based analysis.

        Args:
            obj: ARKitScenes object
            annotation_id: SceneFun3D annotation ID

        Returns:
            Tuple of (overlap_ratio, distance, mesh_proximity_score)
        """
        # Get affordance points in SceneFun3D coordinates (native - no transformation)
        affordance_points = self.get_affordance_points_in_scenefun3d(annotation_id)

        if len(affordance_points) == 0:
            return 0.0, float('inf'), 0.0

        # Transform ARKit object to SceneFun3D coordinates
        object_center, object_min, object_max = self.transform_arkit_object_to_scenefun3d(obj)

        # Compute basic geometric overlap
        inside_bbox = np.all((affordance_points >= object_min) &
                             (affordance_points <= object_max), axis=1)
        overlap_ratio = np.sum(inside_bbox) / len(affordance_points)

        # Compute distance between centers
        affordance_center = np.mean(affordance_points, axis=0)
        distance = np.linalg.norm(affordance_center - object_center)

        # Enhanced mesh-based proximity analysis
        mesh_proximity_score = self.compute_mesh_proximity(obj, affordance_points)

        return overlap_ratio, distance, mesh_proximity_score

    def compute_mesh_proximity(self, obj: ARKitObject, affordance_points: np.ndarray) -> float:
        """Compute mesh-based proximity score using ARKit reconstruction.

        This uses the actual ARKit mesh geometry for more precise spatial analysis.
        """
        try:
            # Load ARKit mesh in point cloud format
            arkit_pc = self.scenefun3d_parser.get_arkit_reconstruction(
                self.visit_id, self.video_id, format="point_cloud"
            )

            if len(arkit_pc.points) == 0:
                return 0.0

            # Transform ARKit mesh points to SceneFun3D coordinates
            arkit_points = np.asarray(arkit_pc.points)
            transformed_arkit_points = self.transformer.transform_arkit_to_scenefun3d(arkit_points)

            # Find closest distances between affordance points and mesh points
            if len(affordance_points) > 0 and len(transformed_arkit_points) > 0:
                # Use subset for efficiency (sample 1000 points max)
                n_afford = min(len(affordance_points), 1000)
                n_mesh = min(len(transformed_arkit_points), 5000)

                afford_sample = affordance_points[:n_afford]
                mesh_sample = transformed_arkit_points[:n_mesh]

                # Compute pairwise distances (efficient approach)
                distances = np.linalg.norm(
                    afford_sample[:, np.newaxis, :] - mesh_sample[np.newaxis, :, :],
                    axis=2
                )
                min_distances = np.min(distances, axis=1)

                # Proximity score based on how close affordance points are to mesh
                # Higher score for points closer to the mesh surface
                proximity_threshold = 50.0  # mm
                close_points = min_distances < proximity_threshold
                proximity_score = np.sum(close_points) / len(min_distances)

                return proximity_score

        except Exception as e:
            print(f"   ⚠️ Mesh proximity computation failed for {obj.label}: {e}")

        return 0.0

    def compute_geometric_plausibility(self, obj: ARKitObject, annotation_id: str, task_description: str) -> float:
        """Compute geometric plausibility score based on object type and task."""
        # Basic semantic filtering
        semantic_matches = {
            'toilet': ['flush', 'toilet'],
            'sink': ['tap', 'faucet', 'sink'],
            'bathtub': ['bathtub', 'tub', 'bath'],
            'door': ['door', 'close', 'open'],
            'window': ['window'],
            'mirror': ['mirror']
        }

        object_label = obj.label.lower()
        task_lower = task_description.lower()

        # Check for semantic compatibility
        if object_label in semantic_matches:
            keywords = semantic_matches[object_label]
            if any(keyword in task_lower for keyword in keywords):
                return 1.0  # High plausibility for semantic match

        # Penalize obvious mismatches
        obvious_mismatches = [
            ('toilet', 'sink'),
            ('toilet', 'window'),
            ('toilet', 'door'),
            ('toilet', 'mirror'),
            ('sink', 'toilet'),
            ('sink', 'door'),
            ('bathtub', 'toilet'),
            ('bathtub', 'sink')
        ]

        for obj_type, task_keyword in obvious_mismatches:
            if obj_type in object_label and task_keyword in task_lower:
                return 0.0  # Zero plausibility for obvious mismatch

        return 0.5  # Neutral plausibility

    def find_confident_object_for_affordance(self, annotation_id: str, task_description: str,
                                           confidence_threshold: float = 0.3) -> Optional[ARKitObject]:
        """Find confident object match for affordance - NO FALLBACK.

        This is the KEY CHANGE - if no confident match is found, return None instead of fallback.

        Args:
            annotation_id: SceneFun3D annotation ID
            task_description: Natural language task description
            confidence_threshold: Minimum confidence for valid match

        Returns:
            ARKitObject if confident match found, None otherwise
        """
        best_object = None
        best_confidence = 0.0
        best_relationship = None

        for obj in self.arkitscenes_parser.get_objects():
            # Compute enhanced spatial relationship
            overlap_ratio, distance, mesh_proximity = self.compute_enhanced_overlap(obj, annotation_id)

            # Compute geometric plausibility
            geometric_plausibility = self.compute_geometric_plausibility(obj, annotation_id, task_description)

            # Combined confidence score
            # Weight: overlap (40%), proximity to mesh (30%), geometric plausibility (30%)
            confidence = (overlap_ratio * 0.4 +
                         mesh_proximity * 0.3 +
                         geometric_plausibility * 0.3)

            print(f"   📊 {obj.label} vs '{task_description}': overlap={overlap_ratio:.3f}, "
                  f"mesh_prox={mesh_proximity:.3f}, geom_plaus={geometric_plausibility:.3f}, "
                  f"confidence={confidence:.3f}")

            if confidence > best_confidence:
                best_confidence = confidence
                best_object = obj
                best_relationship = EnhancedSpatialRelationship(
                    object_id=obj.uid,
                    object_label=obj.label,
                    annotation_id=annotation_id,
                    task_description=task_description,
                    overlap_ratio=overlap_ratio,
                    distance=distance,
                    confidence=confidence,
                    affordance_type="Interact",  # Will be inferred from task
                    mesh_proximity_score=mesh_proximity,
                    geometric_plausibility=geometric_plausibility
                )

        # CRITICAL: Only return if confidence exceeds threshold
        if best_relationship and best_relationship.is_confident_match(confidence_threshold):
            print(f"   ✅ Confident match found: {best_object.label} (confidence: {best_confidence:.3f})")
            return best_object
        else:
            print(f"   ❌ No confident match found (best: {best_confidence:.3f} < {confidence_threshold})")
            return None

    def analyze_all_enhanced_relationships(self, confidence_threshold: float = 0.3) -> List[EnhancedSpatialRelationship]:
        """Analyze all spatial relationships with enhanced methodology.

        Args:
            confidence_threshold: Minimum confidence for including relationships

        Returns:
            List of confident spatial relationships only
        """
        print(f"\n🔍 Enhanced Spatial Analysis (confidence threshold: {confidence_threshold})")
        print("=" * 60)

        self.load_scenefun3d_data()
        relationships = []

        for desc in self.descriptions:
            task_description = desc['description']
            annotation_ids = desc['annot_id']

            print(f"\n📋 Task: '{task_description}'")
            print(f"   📍 Annotation IDs: {annotation_ids}")

            for annotation_id in annotation_ids:
                if annotation_id not in self.annotation_lookup:
                    print(f"   ⚠️ Annotation {annotation_id} not found in annotation data")
                    continue

                # Find confident object match (NO FALLBACK)
                confident_object = self.find_confident_object_for_affordance(
                    annotation_id, task_description, confidence_threshold
                )

                if confident_object:
                    # Create confident relationship
                    overlap_ratio, distance, mesh_proximity = self.compute_enhanced_overlap(
                        confident_object, annotation_id
                    )
                    geometric_plausibility = self.compute_geometric_plausibility(
                        confident_object, annotation_id, task_description
                    )

                    confidence = (overlap_ratio * 0.4 +
                                 mesh_proximity * 0.3 +
                                 geometric_plausibility * 0.3)

                    motion_type = None
                    if annotation_id in self.motion_lookup:
                        motion_type = self.motion_lookup[annotation_id]['motion_type']

                    relationship = EnhancedSpatialRelationship(
                        object_id=confident_object.uid,
                        object_label=confident_object.label,
                        annotation_id=annotation_id,
                        task_description=task_description,
                        overlap_ratio=overlap_ratio,
                        distance=distance,
                        confidence=confidence,
                        affordance_type=self.infer_affordance_type(task_description),
                        motion_type=motion_type,
                        mesh_proximity_score=mesh_proximity,
                        geometric_plausibility=geometric_plausibility
                    )

                    relationships.append(relationship)
                    print(f"   ✅ Added confident relationship: {confident_object.label}")

                else:
                    print(f"   🔍 Affordance {annotation_id} will remain floating (no confident object match)")

        print(f"\n📊 Analysis complete: {len(relationships)} confident relationships found")
        return relationships

    def infer_affordance_type(self, task_description: str) -> str:
        """Infer affordance type from task description."""
        task_lower = task_description.lower()

        if any(word in task_lower for word in ['flush', 'push', 'press']):
            return 'Push'
        elif any(word in task_lower for word in ['turn', 'rotate', 'twist']):
            return 'Rotate'
        elif any(word in task_lower for word in ['pull', 'unplug']):
            return 'Pull'
        elif any(word in task_lower for word in ['open', 'close']):
            return 'Rotate'
        else:
            return 'Interact'


def main():
    """Test enhanced spatial analysis."""
    print("🧪 Enhanced Spatial Analysis Test")
    print("=" * 50)

    # Initialize enhanced analyzer
    data_root = "/nas/jiachen/SceneFun3D/alignment/data_examples/scenefun3d"
    analyzer = EnhancedSpatialAnalyzer(data_root, "422203", "42445781")

    # Run enhanced analysis with strict confidence threshold
    relationships = analyzer.analyze_all_enhanced_relationships(confidence_threshold=0.3)

    # Report results
    print(f"\n📈 Enhanced Analysis Results:")
    print(f"   📊 Total confident relationships: {len(relationships)}")

    confident_count = sum(1 for rel in relationships if rel.is_confident_match(0.3))
    print(f"   📊 High-confidence matches: {confident_count}")

    # Group by object
    object_groups = {}
    for rel in relationships:
        if rel.object_label not in object_groups:
            object_groups[rel.object_label] = []
        object_groups[rel.object_label].append(rel)

    for obj_label, rels in object_groups.items():
        print(f"\n🏷️ {obj_label.upper()} ({len(rels)} relationships):")
        for rel in rels:
            print(f"   📋 '{rel.task_description}' (conf: {rel.confidence:.3f})")


if __name__ == "__main__":
    main()