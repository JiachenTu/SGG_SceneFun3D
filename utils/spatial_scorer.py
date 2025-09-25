"""
Spatial Relationship Scorer - V2.0 Pipeline

Simple but effective confidence calculation for object-affordance spatial relationships.
Implements the key principle: NO fallback logic - only confident matches are accepted.

Key Features:
- Simple overlap-based scoring
- Distance-based confidence
- No fallback connections
- Support for floating affordances
- Clear confidence thresholds
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging


@dataclass
class SpatialRelationship:
    """Container for spatial relationship data."""
    object_id: str
    object_class: str
    affordance_id: str
    affordance_type: str

    # Confidence components
    overlap_ratio: float = 0.0
    distance_score: float = 0.0
    final_confidence: float = 0.0

    # Geometry data
    object_center: Optional[np.ndarray] = None
    object_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    affordance_center: Optional[np.ndarray] = None
    affordance_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None

    # Metadata
    affordance_point_count: int = 0
    is_confident: bool = False
    notes: List[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []


class SpatialScorer:
    """
    Simple spatial relationship scorer with no fallback logic.

    Calculates confidence scores based on:
    1. Overlap ratio (60%): How much affordance overlaps with object bounding box
    2. Distance score (40%): Inverse distance between centers
    """

    def __init__(self, confidence_threshold: float = 0.2,
                 distance_weight: float = 0.4,
                 overlap_weight: float = 0.6,
                 max_distance: float = 0.5,  # 0.5m in meters
                 voxel_size: float = 0.005,  # 5mm laser scan resolution
                 logger: Optional[logging.Logger] = None):
        """
        Initialize spatial scorer.

        Args:
            confidence_threshold: Minimum confidence for accepting relationships
            distance_weight: Weight for distance component (0-1)
            overlap_weight: Weight for overlap component (0-1)
            max_distance: Maximum distance for scoring (in meters)
            voxel_size: Laser scan voxel resolution (5mm = 0.005m)
            logger: Optional logger
        """
        if abs(distance_weight + overlap_weight - 1.0) > 1e-6:
            raise ValueError("Distance and overlap weights must sum to 1.0")

        self.confidence_threshold = confidence_threshold
        self.distance_weight = distance_weight
        self.overlap_weight = overlap_weight
        self.max_distance = max_distance
        self.voxel_size = voxel_size
        # Spatial uncertainty from voxel sampling: sqrt(3) * (voxel_size/2)
        self.spatial_uncertainty = np.sqrt(3) * (voxel_size / 2)
        self.logger = logger or logging.getLogger(__name__)

        self.logger.info(f"✅ Spatial scorer initialized")
        self.logger.info(f"   🎯 Confidence threshold: {confidence_threshold}")
        self.logger.info(f"   ⚖️ Weights: overlap={overlap_weight}, distance={distance_weight}")
        self.logger.info(f"   📐 Max distance: {max_distance}m")
        self.logger.info(f"   🔲 Voxel size: {voxel_size*1000:.1f}mm")
        self.logger.info(f"   📊 Spatial uncertainty: ±{self.spatial_uncertainty*1000:.1f}mm")

    def calculate_bbox_overlap(self, bbox1: Tuple[np.ndarray, np.ndarray],
                              bbox2: Tuple[np.ndarray, np.ndarray]) -> float:
        """
        Calculate overlap ratio between two axis-aligned bounding boxes.

        Args:
            bbox1: (min_bounds, max_bounds) for first box
            bbox2: (min_bounds, max_bounds) for second box

        Returns:
            Overlap ratio (0-1), where 1 means complete overlap
        """
        min1, max1 = bbox1
        min2, max2 = bbox2

        # Calculate intersection bounds
        intersection_min = np.maximum(min1, min2)
        intersection_max = np.minimum(max1, max2)

        # Check if there's any intersection
        if np.any(intersection_min >= intersection_max):
            return 0.0

        # Calculate volumes
        intersection_volume = np.prod(intersection_max - intersection_min)
        bbox1_volume = np.prod(max1 - min1)
        bbox2_volume = np.prod(max2 - min2)

        # Overlap ratio relative to smaller bounding box
        smaller_volume = min(bbox1_volume, bbox2_volume)
        if smaller_volume <= 0:
            return 0.0

        overlap_ratio = intersection_volume / smaller_volume
        return min(overlap_ratio, 1.0)  # Clamp to [0, 1]

    def calculate_distance_score(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Calculate distance-based confidence score with voxel uncertainty handling.

        Args:
            point1: First 3D point (e.g., object center)
            point2: Second 3D point (e.g., affordance center)

        Returns:
            Distance score (0-1), where 1 means very close, 0 means very far
        """
        raw_distance = np.linalg.norm(point1 - point2)

        # Account for spatial uncertainty from 5mm voxel sampling
        # Reduce effective distance by uncertainty to avoid penalizing voxel sampling noise
        adjusted_distance = max(0, raw_distance - self.spatial_uncertainty)

        # Normalize distance to [0, 1] range
        normalized_distance = min(adjusted_distance / self.max_distance, 1.0)

        # Convert to score (closer = higher score)
        distance_score = 1.0 - normalized_distance

        return distance_score

    def calculate_affordance_bounds(self, affordance_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate axis-aligned bounding box for affordance points with voxel grid alignment.

        Args:
            affordance_points: Nx3 array of 3D points

        Returns:
            Tuple of (min_bounds, max_bounds) aligned to voxel grid
        """
        if len(affordance_points) == 0:
            return np.zeros(3), np.zeros(3)

        raw_min_bounds = np.min(affordance_points, axis=0)
        raw_max_bounds = np.max(affordance_points, axis=0)

        # Quantize bounds to voxel grid to reflect true 5mm resolution
        # Floor min bounds, ceil max bounds to ensure all points are included
        min_bounds = np.floor(raw_min_bounds / self.voxel_size) * self.voxel_size
        max_bounds = np.ceil(raw_max_bounds / self.voxel_size) * self.voxel_size

        # Ensure minimum size of one voxel
        size = max_bounds - min_bounds
        min_size = np.array([self.voxel_size, self.voxel_size, self.voxel_size])
        corrected_size = np.maximum(size, min_size)

        # Adjust bounds to maintain center while enforcing minimum size
        center = (min_bounds + max_bounds) / 2
        min_bounds = center - corrected_size / 2
        max_bounds = center + corrected_size / 2

        return min_bounds, max_bounds

    def calculate_confidence(self, object_center: np.ndarray,
                           object_bounds: Tuple[np.ndarray, np.ndarray],
                           affordance_points: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Calculate spatial relationship confidence.

        Args:
            object_center: 3D center of object
            object_bounds: (min_bounds, max_bounds) of object
            affordance_points: Nx3 array of affordance points

        Returns:
            Tuple of (final_confidence, component_scores)
        """
        if len(affordance_points) == 0:
            return 0.0, {"overlap": 0.0, "distance": 0.0}

        # Calculate affordance bounding box and center
        affordance_bounds = self.calculate_affordance_bounds(affordance_points)
        affordance_center = np.mean(affordance_points, axis=0)

        # Calculate overlap score
        overlap_ratio = self.calculate_bbox_overlap(object_bounds, affordance_bounds)

        # Calculate distance score
        distance_score = self.calculate_distance_score(object_center, affordance_center)

        # Calculate final confidence
        final_confidence = (self.overlap_weight * overlap_ratio +
                          self.distance_weight * distance_score)

        component_scores = {
            "overlap": overlap_ratio,
            "distance": distance_score,
            "final": final_confidence
        }

        return final_confidence, component_scores

    def score_relationship(self, object_id: str, object_class: str,
                          object_center: np.ndarray, object_bounds: Tuple[np.ndarray, np.ndarray],
                          affordance_id: str, affordance_type: str,
                          affordance_points: np.ndarray) -> SpatialRelationship:
        """
        Score a single object-affordance spatial relationship.

        Args:
            object_id: Object identifier
            object_class: Object semantic class
            object_center: 3D center of object
            object_bounds: (min_bounds, max_bounds) of object
            affordance_id: Affordance identifier
            affordance_type: Affordance type/description
            affordance_points: Nx3 array of affordance points

        Returns:
            SpatialRelationship with confidence scores
        """
        # Calculate confidence
        final_confidence, components = self.calculate_confidence(
            object_center, object_bounds, affordance_points
        )

        # Calculate additional metrics
        affordance_center = np.mean(affordance_points, axis=0) if len(affordance_points) > 0 else np.zeros(3)
        affordance_bounds = self.calculate_affordance_bounds(affordance_points)

        # Determine if relationship is confident
        is_confident = final_confidence >= self.confidence_threshold

        # Create relationship
        relationship = SpatialRelationship(
            object_id=object_id,
            object_class=object_class,
            affordance_id=affordance_id,
            affordance_type=affordance_type,
            overlap_ratio=components["overlap"],
            distance_score=components["distance"],
            final_confidence=final_confidence,
            object_center=object_center,
            object_bounds=object_bounds,
            affordance_center=affordance_center,
            affordance_bounds=affordance_bounds,
            affordance_point_count=len(affordance_points),
            is_confident=is_confident
        )

        # Add notes
        if final_confidence < 0.1:
            relationship.notes.append("Very low confidence - likely unrelated")
        elif final_confidence < self.confidence_threshold:
            relationship.notes.append("Below confidence threshold - will be floating")
        elif final_confidence > 0.7:
            relationship.notes.append("High confidence relationship")

        return relationship

    def score_all_relationships(self, objects_data: List[Dict[str, Any]],
                              affordances_data: List[Dict[str, Any]]) -> List[SpatialRelationship]:
        """
        Score all possible object-affordance relationships.

        Args:
            objects_data: List of object dictionaries with center, bounds, etc.
            affordances_data: List of affordance dictionaries with points, etc.

        Returns:
            List of SpatialRelationship objects, sorted by confidence
        """
        relationships = []

        self.logger.info(f"🔍 Scoring {len(objects_data)} objects × {len(affordances_data)} affordances")

        for obj_data in objects_data:
            for aff_data in affordances_data:
                try:
                    # Extract object data
                    obj_id = obj_data.get('id', 'unknown_object')
                    obj_class = obj_data.get('semantic_class', 'unknown')
                    obj_center = np.array(obj_data['center'])
                    obj_bounds = (np.array(obj_data['min_bounds']), np.array(obj_data['max_bounds']))

                    # Extract affordance data
                    aff_id = obj_data.get('id', 'unknown_affordance')
                    aff_type = aff_data.get('type', 'unknown')
                    aff_points = np.array(aff_data['points'])

                    # Score relationship
                    relationship = self.score_relationship(
                        obj_id, obj_class, obj_center, obj_bounds,
                        aff_id, aff_type, aff_points
                    )

                    relationships.append(relationship)

                except Exception as e:
                    self.logger.warning(f"Failed to score {obj_data.get('id', '?')} ↔ {aff_data.get('id', '?')}: {e}")

        # Sort by confidence (highest first)
        relationships.sort(key=lambda r: r.final_confidence, reverse=True)

        # Log statistics
        confident_count = sum(1 for r in relationships if r.is_confident)
        self.logger.info(f"📊 Scored relationships: {len(relationships)} total, {confident_count} confident")

        return relationships

    def find_best_matches(self, relationships: List[SpatialRelationship],
                         allow_multiple_objects_per_affordance: bool = False) -> Dict[str, Optional[str]]:
        """
        Find the best object match for each affordance.

        Args:
            relationships: List of scored spatial relationships
            allow_multiple_objects_per_affordance: Whether one affordance can match multiple objects

        Returns:
            Dictionary mapping affordance_id -> best_object_id (or None for floating)
        """
        matches = {}
        used_objects = set()

        # Group by affordance
        affordance_relationships = {}
        for rel in relationships:
            if rel.affordance_id not in affordance_relationships:
                affordance_relationships[rel.affordance_id] = []
            affordance_relationships[rel.affordance_id].append(rel)

        # Find best match for each affordance
        for aff_id, aff_rels in affordance_relationships.items():
            # Sort by confidence
            aff_rels.sort(key=lambda r: r.final_confidence, reverse=True)

            best_match = None
            for rel in aff_rels:
                # Must be confident
                if not rel.is_confident:
                    continue

                # Check if object is already used (if not allowing multiple matches)
                if not allow_multiple_objects_per_affordance and rel.object_id in used_objects:
                    continue

                # Found best match
                best_match = rel.object_id
                used_objects.add(rel.object_id)
                break

            matches[aff_id] = best_match  # None if no confident match found

        # Log statistics
        matched_count = sum(1 for match in matches.values() if match is not None)
        floating_count = len(matches) - matched_count

        self.logger.info(f"🎯 Matching results: {matched_count} matched, {floating_count} floating")

        return matches

    def get_confidence_statistics(self, relationships: List[SpatialRelationship]) -> Dict[str, Any]:
        """Get statistics about confidence scores."""
        if not relationships:
            return {"count": 0}

        confidences = [r.final_confidence for r in relationships]
        overlap_ratios = [r.overlap_ratio for r in relationships]
        distance_scores = [r.distance_score for r in relationships]

        return {
            "count": len(relationships),
            "confidence": {
                "mean": np.mean(confidences),
                "std": np.std(confidences),
                "min": np.min(confidences),
                "max": np.max(confidences),
                "confident_count": sum(1 for r in relationships if r.is_confident)
            },
            "overlap": {
                "mean": np.mean(overlap_ratios),
                "std": np.std(overlap_ratios),
                "min": np.min(overlap_ratios),
                "max": np.max(overlap_ratios)
            },
            "distance": {
                "mean": np.mean(distance_scores),
                "std": np.std(distance_scores),
                "min": np.min(distance_scores),
                "max": np.max(distance_scores)
            }
        }