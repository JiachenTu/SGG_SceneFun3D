"""
ARKitScenes Data Parser

Utilities for parsing ARKitScenes 3DOD annotations and extracting 3D object information.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class OrientedBoundingBox:
    """Represents an oriented 3D bounding box."""
    centroid: np.ndarray  # [x, y, z] in mm
    axes_lengths: np.ndarray  # [width, height, depth] in mm
    normalized_axes: np.ndarray  # 3x3 rotation matrix (flattened)

    def get_rotation_matrix(self) -> np.ndarray:
        """Get the 3x3 rotation matrix."""
        return self.normalized_axes.reshape(3, 3)

    def get_corners(self) -> np.ndarray:
        """Get the 8 corners of the bounding box."""
        # Half extents
        half_extents = self.axes_lengths / 2

        # Local corners (before rotation)
        local_corners = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ]) * half_extents

        # Apply rotation
        rotation = self.get_rotation_matrix()
        rotated_corners = local_corners @ rotation.T

        # Translate to world position
        world_corners = rotated_corners + self.centroid

        return world_corners


@dataclass
class ARKitObject:
    """Represents a 3D object from ARKitScenes."""
    uid: str
    label: str
    object_id: int
    obb: OrientedBoundingBox
    obb_aligned: Optional[OrientedBoundingBox]
    segments: List[int]
    attributes: Dict
    hierarchy: int

    def get_volume(self) -> float:
        """Calculate the volume of the bounding box in cubic mm."""
        return np.prod(self.obb.axes_lengths)

    def get_size_category(self) -> str:
        """Categorize object by size."""
        volume = self.get_volume()
        if volume < 1e7:  # < 0.01 m³
            return "small"
        elif volume < 1e8:  # < 0.1 m³
            return "medium"
        else:
            return "large"


class ARKitScenesParser:
    """Parser for ARKitScenes 3DOD annotation files."""

    def __init__(self, annotation_file: str):
        """Initialize parser with annotation file path."""
        self.annotation_file = annotation_file
        self.data = None
        self.objects = []

    def load(self) -> Dict:
        """Load and parse the annotation file."""
        with open(self.annotation_file, 'r') as f:
            self.data = json.load(f)

        self._parse_objects()
        return self.data

    def _parse_objects(self) -> None:
        """Parse objects from the loaded data."""
        if not self.data:
            raise ValueError("Data not loaded. Call load() first.")

        self.objects = []
        for obj_data in self.data.get('data', []):
            # Parse main OBB
            obb_data = obj_data['segments']['obb']
            obb = OrientedBoundingBox(
                centroid=np.array(obb_data['centroid']),
                axes_lengths=np.array(obb_data['axesLengths']),
                normalized_axes=np.array(obb_data['normalizedAxes'])
            )

            # Parse aligned OBB if available
            obb_aligned = None
            if 'obbAligned' in obj_data['segments']:
                obb_aligned_data = obj_data['segments']['obbAligned']
                obb_aligned = OrientedBoundingBox(
                    centroid=np.array(obb_aligned_data['centroid']),
                    axes_lengths=np.array(obb_aligned_data['axesLengths']),
                    normalized_axes=np.array(obb_aligned_data['normalizedAxes'])
                )

            # Create object
            obj = ARKitObject(
                uid=obj_data['uid'],
                label=obj_data['label'],
                object_id=obj_data['objectId'],
                obb=obb,
                obb_aligned=obb_aligned,
                segments=obj_data['segments'].get('segments', []),
                attributes=obj_data.get('attributes', {}),
                hierarchy=obj_data.get('hierarchy', 1)
            )

            self.objects.append(obj)

    def get_objects(self) -> List[ARKitObject]:
        """Get list of parsed objects."""
        return self.objects

    def get_objects_by_label(self, label: str) -> List[ARKitObject]:
        """Get objects filtered by label."""
        return [obj for obj in self.objects if obj.label == label]

    def get_scene_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the overall scene bounding box."""
        if not self.objects:
            return np.zeros(3), np.zeros(3)

        all_corners = []
        for obj in self.objects:
            corners = obj.obb.get_corners()
            all_corners.append(corners)

        all_corners = np.vstack(all_corners)
        min_bounds = np.min(all_corners, axis=0)
        max_bounds = np.max(all_corners, axis=0)

        return min_bounds, max_bounds

    def get_stats(self) -> Dict:
        """Get statistics about the parsed data."""
        if not self.data:
            return {}

        stats = self.data.get('stats', {})
        stats.update({
            'parsed_objects': len(self.objects),
            'object_labels': [obj.label for obj in self.objects],
            'unique_labels': list(set(obj.label for obj in self.objects))
        })

        return stats

    def print_summary(self) -> None:
        """Print a summary of the parsed data."""
        print(f"ARKitScenes Annotation Summary:")
        print(f"File: {self.annotation_file}")
        print(f"Objects detected: {len(self.objects)}")

        for obj in self.objects:
            print(f"\n{obj.label.upper()} (ID: {obj.uid[:8]}...)")
            print(f"  Center: [{obj.obb.centroid[0]:.1f}, {obj.obb.centroid[1]:.1f}, {obj.obb.centroid[2]:.1f}] mm")
            print(f"  Size: [{obj.obb.axes_lengths[0]:.1f} × {obj.obb.axes_lengths[1]:.1f} × {obj.obb.axes_lengths[2]:.1f}] mm")
            print(f"  Volume: {obj.get_volume():.0f} mm³ ({obj.get_size_category()})")
            print(f"  Segments: {len(obj.segments)}")

        min_bounds, max_bounds = self.get_scene_bounds()
        scene_size = max_bounds - min_bounds
        print(f"\nScene bounds: [{scene_size[0]:.0f} × {scene_size[1]:.0f} × {scene_size[2]:.0f}] mm")


def main():
    """Example usage of the ARKitScenes parser."""
    annotation_file = "/home/jiachen/scratch/SceneFun3D/alignment/data_examples/arkitscenes/video_42445781/42445781_3dod_annotation.json"

    parser = ARKitScenesParser(annotation_file)
    parser.load()
    parser.print_summary()

    # Example: Get toilet object
    toilets = parser.get_objects_by_label("toilet")
    if toilets:
        toilet = toilets[0]
        print(f"\nToilet corners (8 points):")
        corners = toilet.obb.get_corners()
        for i, corner in enumerate(corners):
            print(f"  Corner {i}: [{corner[0]:.1f}, {corner[1]:.1f}, {corner[2]:.1f}]")


if __name__ == "__main__":
    main()