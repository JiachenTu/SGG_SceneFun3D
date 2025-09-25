"""
SceneFun3D Data Parser

Utilities for parsing SceneFun3D annotations, descriptions, and motion data.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Motion:
    """Represents a motion parameter."""
    motion_id: str
    annot_id: str
    motion_type: str  # "rot" or "trans"
    motion_dir: np.ndarray  # 3D direction vector
    motion_origin_idx: int  # Point index in laser scan
    motion_viz_orient: str  # "inwards" or "outwards"

    def is_rotational(self) -> bool:
        """Check if motion is rotational."""
        return self.motion_type == "rot"

    def is_translational(self) -> bool:
        """Check if motion is translational."""
        return self.motion_type == "trans"


@dataclass
class Annotation:
    """Represents an affordance annotation."""
    annot_id: str
    indices: List[int]  # Point indices in laser scan
    motion: Optional[Motion] = None

    def get_point_count(self) -> int:
        """Get number of annotated points."""
        return len(self.indices)

    def get_bounds(self, points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box of annotated points."""
        if not self.indices:
            return np.zeros(3), np.zeros(3)

        selected_points = points_3d[self.indices]
        min_bounds = np.min(selected_points, axis=0)
        max_bounds = np.max(selected_points, axis=0)

        return min_bounds, max_bounds

    def get_centroid(self, points_3d: np.ndarray) -> np.ndarray:
        """Get centroid of annotated points."""
        if not self.indices:
            return np.zeros(3)

        selected_points = points_3d[self.indices]
        return np.mean(selected_points, axis=0)


@dataclass
class TaskDescription:
    """Represents a task description with associated annotations."""
    desc_id: str
    description: str
    annot_ids: List[str]
    annotations: List[Annotation]

    def get_primary_annotation(self) -> Optional[Annotation]:
        """Get the primary (first) annotation for this task."""
        return self.annotations[0] if self.annotations else None

    def infer_affordance_type(self) -> str:
        """Infer affordance type from task description."""
        desc_lower = self.description.lower()

        # Rotational affordances
        if any(word in desc_lower for word in ['turn', 'rotate', 'twist', 'spin']):
            return "Rotate"
        elif any(word in desc_lower for word in ['open', 'close']):
            return "Rotate"  # Most open/close are rotational

        # Translational affordances
        elif any(word in desc_lower for word in ['push', 'press', 'push down']):
            return "Push"
        elif any(word in desc_lower for word in ['pull', 'unplug', 'pull out']):
            return "Pull"
        elif any(word in desc_lower for word in ['lift', 'raise']):
            return "Lift"

        # Default
        return "Interact"

    def infer_target_object(self) -> Optional[str]:
        """Infer target object from task description."""
        desc_lower = self.description.lower()

        # Common bathroom objects
        if 'toilet' in desc_lower:
            return "toilet"
        elif 'sink' in desc_lower or 'tap' in desc_lower:
            return "sink"
        elif 'bathtub' in desc_lower or 'bath' in desc_lower:
            return "bathtub"
        elif 'door' in desc_lower:
            return "door"
        elif 'window' in desc_lower:
            return "window"
        elif 'mirror' in desc_lower:
            return "mirror"

        return None


class SceneFun3DParser:
    """Parser for SceneFun3D annotation files."""

    def __init__(self, data_dir: str):
        """Initialize parser with data directory."""
        self.data_dir = data_dir
        self.visit_id = None
        self.descriptions = []
        self.annotations = {}
        self.motions = {}

    def load(self, visit_id: str) -> None:
        """Load all data files for a visit."""
        self.visit_id = visit_id

        # File paths
        desc_file = f"{self.data_dir}/{visit_id}_descriptions.json"
        annot_file = f"{self.data_dir}/{visit_id}_annotations.json"
        motion_file = f"{self.data_dir}/{visit_id}_motions.json"

        # Load descriptions
        with open(desc_file, 'r') as f:
            desc_data = json.load(f)
            self._parse_descriptions(desc_data)

        # Load annotations
        with open(annot_file, 'r') as f:
            annot_data = json.load(f)
            self._parse_annotations(annot_data)

        # Load motions
        with open(motion_file, 'r') as f:
            motion_data = json.load(f)
            self._parse_motions(motion_data)

        # Link motions to annotations
        self._link_motions_to_annotations()

    def _parse_descriptions(self, data: Dict) -> None:
        """Parse task descriptions."""
        self.descriptions = []

        for desc_data in data.get('descriptions', []):
            task = TaskDescription(
                desc_id=desc_data['desc_id'],
                description=desc_data['description'],
                annot_ids=desc_data['annot_id'],
                annotations=[]  # Will be populated later
            )
            self.descriptions.append(task)

    def _parse_annotations(self, data: Dict) -> None:
        """Parse affordance annotations."""
        self.annotations = {}

        for annot_data in data.get('annotations', []):
            annotation = Annotation(
                annot_id=annot_data['annot_id'],
                indices=annot_data['indices']
            )
            self.annotations[annotation.annot_id] = annotation

    def _parse_motions(self, data: Dict) -> None:
        """Parse motion parameters."""
        self.motions = {}

        for motion_data in data.get('motions', []):
            motion = Motion(
                motion_id=motion_data['motion_id'],
                annot_id=motion_data['annot_id'],
                motion_type=motion_data['motion_type'],
                motion_dir=np.array(motion_data['motion_dir']),
                motion_origin_idx=motion_data['motion_origin_idx'],
                motion_viz_orient=motion_data['motion_viz_orient']
            )
            self.motions[motion.annot_id] = motion

    def _link_motions_to_annotations(self) -> None:
        """Link motion data to annotations and descriptions."""
        # Link motions to annotations
        for annot_id, annotation in self.annotations.items():
            if annot_id in self.motions:
                annotation.motion = self.motions[annot_id]

        # Link annotations to descriptions
        for task in self.descriptions:
            task.annotations = []
            for annot_id in task.annot_ids:
                if annot_id in self.annotations:
                    task.annotations.append(self.annotations[annot_id])

    def get_task_descriptions(self) -> List[TaskDescription]:
        """Get all task descriptions."""
        return self.descriptions

    def get_task_by_description(self, description_text: str) -> Optional[TaskDescription]:
        """Find task by description text (partial match)."""
        desc_lower = description_text.lower()
        for task in self.descriptions:
            if desc_lower in task.description.lower():
                return task
        return None

    def get_annotations_by_motion_type(self, motion_type: str) -> List[Annotation]:
        """Get annotations filtered by motion type."""
        result = []
        for annotation in self.annotations.values():
            if annotation.motion and annotation.motion.motion_type == motion_type:
                result.append(annotation)
        return result

    def get_stats(self) -> Dict:
        """Get statistics about the parsed data."""
        total_points = sum(len(annot.indices) for annot in self.annotations.values())
        motion_types = [motion.motion_type for motion in self.motions.values()]

        return {
            'visit_id': self.visit_id,
            'num_tasks': len(self.descriptions),
            'num_annotations': len(self.annotations),
            'num_motions': len(self.motions),
            'total_annotated_points': total_points,
            'motion_types': {
                'rotational': motion_types.count('rot'),
                'translational': motion_types.count('trans')
            },
            'tasks': [task.description for task in self.descriptions]
        }

    def print_summary(self) -> None:
        """Print a summary of the parsed data."""
        print(f"SceneFun3D Data Summary:")
        print(f"Visit ID: {self.visit_id}")
        print(f"Task descriptions: {len(self.descriptions)}")
        print(f"Annotations: {len(self.annotations)}")
        print(f"Motion parameters: {len(self.motions)}")

        print(f"\nTask Descriptions:")
        for i, task in enumerate(self.descriptions, 1):
            affordance_type = task.infer_affordance_type()
            target_object = task.infer_target_object()
            print(f"  {i}. \"{task.description}\"")
            print(f"     → {affordance_type} on {target_object}")
            print(f"     → {len(task.annotations)} annotation(s)")

        print(f"\nMotion Types:")
        rot_count = len([m for m in self.motions.values() if m.is_rotational()])
        trans_count = len([m for m in self.motions.values() if m.is_translational()])
        print(f"  Rotational: {rot_count}")
        print(f"  Translational: {trans_count}")


def main():
    """Example usage of the SceneFun3D parser."""
    data_dir = "/home/jiachen/scratch/SceneFun3D/alignment/data_examples/scenefun3d/visit_422203"

    parser = SceneFun3DParser(data_dir)
    parser.load("422203")
    parser.print_summary()

    # Example: Get flush toilet task
    flush_task = parser.get_task_by_description("flush")
    if flush_task:
        print(f"\nFlush Toilet Task Analysis:")
        print(f"Description: \"{flush_task.description}\"")
        print(f"Affordance type: {flush_task.infer_affordance_type()}")
        print(f"Target object: {flush_task.infer_target_object()}")

        primary_annot = flush_task.get_primary_annotation()
        if primary_annot and primary_annot.motion:
            motion = primary_annot.motion
            print(f"Motion type: {motion.motion_type}")
            print(f"Motion direction: {motion.motion_dir}")
            print(f"Motion origin index: {motion.motion_origin_idx}")


if __name__ == "__main__":
    main()