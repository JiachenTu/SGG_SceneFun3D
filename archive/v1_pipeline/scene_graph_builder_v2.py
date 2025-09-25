"""
Scene Graph Builder V2.0

Enhanced hierarchical scene graph generation with support for:
- Floating affordances (no parent object)
- Confidence-based relationships
- Integration with unified data loader
- Configurable spatial analysis

Architecture:
Level 0: Scene Root
Level 1: Spatial Regions (optional)
Level 2: Objects (from ARKitScenes, transformed to SceneFun3D)
Level 3: Affordances (from SceneFun3D, with confidence scores)
Level 4: Floating Affordances (unattached to objects)
"""

import sys
import os
import json
import uuid
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils.unified_data_loader import UnifiedDataLoader, SceneData
    from utils.enhanced_coordinate_transform import EnhancedCoordinateTransformer, create_transformer_from_matrix
    from utils.spatial_scorer import SpatialScorer, SpatialRelationship
    from utils.arkitscenes_parser import ARKitScenesParser, ARKitObject
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("   Make sure all utility modules are available")
    sys.exit(1)


@dataclass
class SceneGraphNode:
    """Base scene graph node with enhanced metadata."""
    node_id: str
    node_type: str
    label: str = ""
    confidence: float = 1.0
    coordinate_system: str = "SceneFun3D"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SceneRootNode(SceneGraphNode):
    """Root node representing the entire scene."""
    visit_id: str = ""
    video_id: str = ""
    scene_description: str = ""
    task_description: str = ""
    spatial_bounds: Tuple[np.ndarray, np.ndarray] = None
    object_count: int = 0
    affordance_count: int = 0
    floating_affordance_count: int = 0

    def __post_init__(self):
        super().__post_init__()
        self.node_type = "scene_root"


@dataclass
class ObjectNode(SceneGraphNode):
    """Node representing a physical object (transformed to SceneFun3D coordinates)."""
    semantic_class: str = ""
    center: np.ndarray = None
    size: np.ndarray = None
    rotation: np.ndarray = None
    bounds: Tuple[np.ndarray, np.ndarray] = None
    volume: float = 0.0
    original_arkit_center: np.ndarray = None  # Original ARKit coordinates
    children_affordances: List[str] = None

    def __post_init__(self):
        super().__post_init__()
        self.node_type = "object"
        if self.children_affordances is None:
            self.children_affordances = []


@dataclass
class AffordanceNode(SceneGraphNode):
    """Node representing an affordance with spatial relationship info."""
    affordance_type: str = ""
    annotation_id: str = ""
    motion_type: str = ""
    motion_direction: np.ndarray = None
    motion_origin: np.ndarray = None

    # Spatial relationship data
    parent_object_id: Optional[str] = None
    spatial_confidence: float = 0.0
    point_count: int = 0
    center: np.ndarray = None
    bounds: Tuple[np.ndarray, np.ndarray] = None

    # Status
    is_floating: bool = False
    relationship_notes: List[str] = None

    def __post_init__(self):
        super().__post_init__()
        self.node_type = "affordance"
        if self.relationship_notes is None:
            self.relationship_notes = []


class SceneGraphV2:
    """Container for the complete scene graph."""

    def __init__(self, task_description: str, visit_id: str, video_id: str):
        """Initialize scene graph."""
        self.task_description = task_description
        self.visit_id = visit_id
        self.video_id = video_id

        # Nodes
        self.root_node: Optional[SceneRootNode] = None
        self.object_nodes: Dict[str, ObjectNode] = {}
        self.affordance_nodes: Dict[str, AffordanceNode] = {}

        # Relationships
        self.object_affordance_relationships: Dict[str, List[str]] = {}  # object_id -> affordance_ids
        self.floating_affordances: List[str] = []

        # Metadata
        self.creation_timestamp = None
        self.confidence_statistics: Dict[str, Any] = {}

    def add_root_node(self, scene_description: str = "", spatial_bounds: Tuple[np.ndarray, np.ndarray] = None):
        """Add root node to scene graph."""
        self.root_node = SceneRootNode(
            node_id="scene_root",
            label=scene_description or f"Scene {self.visit_id}",
            visit_id=self.visit_id,
            video_id=self.video_id,
            scene_description=scene_description,
            task_description=self.task_description,
            spatial_bounds=spatial_bounds
        )

    def add_object_node(self, object_data: Dict[str, Any]) -> str:
        """Add object node to scene graph."""
        object_id = object_data.get('id', f"object_{len(self.object_nodes)}")

        node = ObjectNode(
            node_id=object_id,
            label=f"{object_data.get('semantic_class', 'object')} ({object_id})",
            semantic_class=object_data.get('semantic_class', 'unknown'),
            center=np.array(object_data.get('center', [0, 0, 0])),
            size=np.array(object_data.get('size', [1, 1, 1])),
            rotation=np.array(object_data.get('rotation', np.eye(3))),
            bounds=(
                np.array(object_data.get('min_bounds', [0, 0, 0])),
                np.array(object_data.get('max_bounds', [1, 1, 1]))
            ),
            volume=object_data.get('volume', 0.0),
            original_arkit_center=np.array(object_data.get('original_center', [0, 0, 0])),
            confidence=1.0  # Objects are always confident
        )

        self.object_nodes[object_id] = node
        self.object_affordance_relationships[object_id] = []

        return object_id

    def add_affordance_node(self, affordance_data: Dict[str, Any],
                          spatial_relationship: Optional[SpatialRelationship] = None) -> str:
        """Add affordance node to scene graph."""
        affordance_id = affordance_data.get('annotation_id', f"affordance_{len(self.affordance_nodes)}")

        # Determine parent and floating status
        parent_object_id = None
        is_floating = True
        spatial_confidence = 0.0
        relationship_notes = []

        if spatial_relationship and spatial_relationship.is_confident:
            parent_object_id = spatial_relationship.object_id
            is_floating = False
            spatial_confidence = spatial_relationship.final_confidence
            relationship_notes = spatial_relationship.notes.copy()

        node = AffordanceNode(
            node_id=affordance_id,
            label=f"{affordance_data.get('type', 'affordance')} ({affordance_id})",
            affordance_type=affordance_data.get('type', 'unknown'),
            annotation_id=affordance_id,
            motion_type=affordance_data.get('motion_type', ''),
            motion_direction=np.array(affordance_data.get('motion_direction', [0, 0, 0])),
            motion_origin=np.array(affordance_data.get('motion_origin', [0, 0, 0])),
            parent_object_id=parent_object_id,
            spatial_confidence=spatial_confidence,
            point_count=affordance_data.get('point_count', 0),
            center=np.array(affordance_data.get('center', [0, 0, 0])),
            bounds=affordance_data.get('bounds', (np.zeros(3), np.zeros(3))),
            is_floating=is_floating,
            relationship_notes=relationship_notes,
            confidence=spatial_confidence
        )

        self.affordance_nodes[affordance_id] = node

        # Update relationships
        if not is_floating and parent_object_id in self.object_affordance_relationships:
            self.object_affordance_relationships[parent_object_id].append(affordance_id)
            # Update object's children list
            if parent_object_id in self.object_nodes:
                self.object_nodes[parent_object_id].children_affordances.append(affordance_id)
        else:
            self.floating_affordances.append(affordance_id)

        return affordance_id

    def update_root_node_statistics(self):
        """Update root node with current statistics."""
        if self.root_node:
            self.root_node.object_count = len(self.object_nodes)
            self.root_node.affordance_count = len(self.affordance_nodes)
            self.root_node.floating_affordance_count = len(self.floating_affordances)

    def to_dict(self) -> Dict[str, Any]:
        """Convert scene graph to dictionary format."""
        # Convert nodes to dictionaries
        def convert_node(node):
            node_dict = asdict(node)
            # Convert numpy arrays to lists
            for key, value in node_dict.items():
                if isinstance(value, np.ndarray):
                    node_dict[key] = value.tolist()
                elif isinstance(value, tuple) and len(value) == 2:
                    # Handle bounds tuples
                    if isinstance(value[0], np.ndarray):
                        node_dict[key] = [value[0].tolist(), value[1].tolist()]
            return node_dict

        return {
            "task_description": self.task_description,
            "visit_id": self.visit_id,
            "video_id": self.video_id,
            "root_node": convert_node(self.root_node) if self.root_node else None,
            "object_nodes": {k: convert_node(v) for k, v in self.object_nodes.items()},
            "affordance_nodes": {k: convert_node(v) for k, v in self.affordance_nodes.items()},
            "object_affordance_relationships": self.object_affordance_relationships,
            "floating_affordances": self.floating_affordances,
            "confidence_statistics": self.confidence_statistics,
            "metadata": {
                "coordinate_system": "SceneFun3D",
                "creation_timestamp": self.creation_timestamp,
                "total_nodes": 1 + len(self.object_nodes) + len(self.affordance_nodes),
                "confident_relationships": sum(
                    1 for node in self.affordance_nodes.values()
                    if not node.is_floating
                )
            }
        }


class SceneGraphBuilderV2:
    """Enhanced scene graph builder with floating affordance support."""

    def __init__(self, confidence_threshold: float = 0.3,
                 logger: Optional[logging.Logger] = None):
        """Initialize scene graph builder."""
        self.confidence_threshold = confidence_threshold
        self.logger = logger or logging.getLogger(__name__)

        # Components
        self.spatial_scorer = SpatialScorer(
            confidence_threshold=confidence_threshold,
            logger=self.logger
        )

    def load_arkit_objects(self, arkitscenes_file: str) -> List[Dict[str, Any]]:
        """Load ARKitScenes objects."""
        try:
            parser = ARKitScenesParser(arkitscenes_file)
            parser.load()
            objects = parser.get_objects()

            object_data = []
            for obj in objects:
                obj_dict = {
                    'id': obj.object_uid or f"arkit_object_{len(object_data)}",
                    'semantic_class': obj.semantic_class,
                    'center': obj.center,
                    'size': obj.size,
                    'rotation': obj.rotation,
                    'volume': np.prod(obj.size) if obj.size is not None else 0.0,
                    'original_center': obj.center.copy(),
                }
                object_data.append(obj_dict)

            self.logger.info(f"✅ Loaded {len(object_data)} ARKit objects")
            return object_data

        except Exception as e:
            self.logger.error(f"Failed to load ARKit objects: {e}")
            return []

    def transform_objects_to_scenefun3d(self, objects_data: List[Dict[str, Any]],
                                      transformer: EnhancedCoordinateTransformer) -> List[Dict[str, Any]]:
        """Transform objects from ARKit to SceneFun3D coordinates."""
        try:
            transformed_objects = transformer.batch_transform_objects(objects_data)

            # Add bounding box information
            for obj in transformed_objects:
                if 'transformation_error' not in obj:
                    center = np.array(obj['center'])
                    size = np.array(obj['size'])
                    rotation = np.array(obj['rotation'])

                    # Calculate axis-aligned bounding box
                    min_bounds, max_bounds = transformer.get_axis_aligned_bbox_from_obb(
                        center, size, rotation
                    )

                    obj['min_bounds'] = min_bounds.tolist()
                    obj['max_bounds'] = max_bounds.tolist()

            successful_transforms = [obj for obj in transformed_objects if 'transformation_error' not in obj]
            self.logger.info(f"✅ Transformed {len(successful_transforms)}/{len(objects_data)} objects to SceneFun3D")

            return successful_transforms

        except Exception as e:
            self.logger.error(f"Failed to transform objects: {e}")
            return []

    def extract_affordances_for_task(self, scene_data: SceneData, task_description: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract affordances relevant to a specific task."""
        affordance_data = []

        try:
            # Get annotation IDs for this task
            task_annotation_ids = task_description.get('annot_ids', [])

            for annot_id in task_annotation_ids:
                # Get affordance points in SceneFun3D coordinates
                affordance_points = scene_data.annotation_lookup.get(annot_id, {}).get('indices', [])
                if not affordance_points:
                    continue

                # Convert indices to 3D points
                if scene_data.laser_scan:
                    laser_points = np.asarray(scene_data.laser_scan.points)
                    valid_indices = [i for i in affordance_points if 0 <= i < len(laser_points)]
                    points_3d = laser_points[valid_indices] if valid_indices else np.array([]).reshape(0, 3)
                else:
                    points_3d = np.array([]).reshape(0, 3)

                # Get motion information
                motion_info = scene_data.motion_lookup.get(annot_id, {})

                # Calculate affordance center and bounds
                if len(points_3d) > 0:
                    center = np.mean(points_3d, axis=0)
                    min_bounds = np.min(points_3d, axis=0)
                    max_bounds = np.max(points_3d, axis=0)
                    bounds = (min_bounds, max_bounds)
                else:
                    center = np.zeros(3)
                    bounds = (np.zeros(3), np.zeros(3))

                affordance_dict = {
                    'annotation_id': annot_id,
                    'type': task_description.get('description', 'unknown_task'),
                    'motion_type': motion_info.get('motion_type', ''),
                    'motion_direction': np.array(motion_info.get('motion_direction', [0, 0, 0])),
                    'motion_origin': np.array(motion_info.get('motion_origin', [0, 0, 0])),
                    'points': points_3d,
                    'point_count': len(points_3d),
                    'center': center,
                    'bounds': bounds
                }

                affordance_data.append(affordance_dict)

            self.logger.info(f"✅ Extracted {len(affordance_data)} affordances for task")
            return affordance_data

        except Exception as e:
            self.logger.error(f"Failed to extract affordances: {e}")
            return []

    def build_scene_graph(self, data_root: str, arkitscenes_file: str,
                         visit_id: str, video_id: str, task_description: Dict[str, Any]) -> SceneGraphV2:
        """
        Build complete scene graph for a single task.

        Args:
            data_root: SceneFun3D data root directory
            arkitscenes_file: Path to ARKitScenes annotation file
            visit_id: Visit identifier
            video_id: Video identifier
            task_description: Task description dictionary

        Returns:
            Complete SceneGraphV2 object
        """
        task_desc = task_description.get('description', 'Unknown task')
        self.logger.info(f"🏗️ Building scene graph for: '{task_desc}'")

        # Initialize scene graph
        scene_graph = SceneGraphV2(task_desc, visit_id, video_id)

        try:
            # 1. Load scene data
            self.logger.info("📂 Loading scene data...")
            loader = UnifiedDataLoader(data_root, visit_id, video_id)
            scene_data = loader.load_all_data(include_mesh=False, include_point_cloud=False)

            if scene_data.load_errors:
                self.logger.warning(f"Scene data loading had {len(scene_data.load_errors)} errors")

            # 2. Load and transform ARKit objects
            self.logger.info("🔄 Loading and transforming ARKit objects...")
            arkit_objects = self.load_arkit_objects(arkitscenes_file)

            if not arkit_objects:
                self.logger.error("No ARKit objects loaded - cannot build scene graph")
                return scene_graph

            # Create coordinate transformer
            transformer = create_transformer_from_matrix(scene_data.transform_matrix)
            transformed_objects = self.transform_objects_to_scenefun3d(arkit_objects, transformer)

            # 3. Extract affordances for this task
            self.logger.info("📍 Extracting task affordances...")
            task_affordances = self.extract_affordances_for_task(scene_data, task_description)

            # 4. Calculate spatial relationships
            self.logger.info("🎯 Calculating spatial relationships...")
            relationships = self.spatial_scorer.score_all_relationships(
                transformed_objects, task_affordances
            )

            # Get best matches for affordances
            matches = self.spatial_scorer.find_best_matches(relationships)

            # 5. Build scene graph
            self.logger.info("🏗️ Assembling scene graph...")

            # Add root node
            if scene_data.laser_scan:
                points = np.asarray(scene_data.laser_scan.points)
                spatial_bounds = (np.min(points, axis=0), np.max(points, axis=0))
            else:
                spatial_bounds = (np.zeros(3), np.ones(3))

            scene_graph.add_root_node(f"Scene {visit_id} - {task_desc}", spatial_bounds)

            # Add object nodes
            for obj_data in transformed_objects:
                scene_graph.add_object_node(obj_data)

            # Add affordance nodes with relationships
            for aff_data in task_affordances:
                aff_id = aff_data['annotation_id']

                # Find best spatial relationship
                best_relationship = None
                for rel in relationships:
                    if rel.affordance_id == aff_id and rel.is_confident:
                        best_relationship = rel
                        break

                scene_graph.add_affordance_node(aff_data, best_relationship)

            # Update statistics
            scene_graph.update_root_node_statistics()
            scene_graph.confidence_statistics = self.spatial_scorer.get_confidence_statistics(relationships)

            # Log results
            confident_count = len([n for n in scene_graph.affordance_nodes.values() if not n.is_floating])
            floating_count = len(scene_graph.floating_affordances)

            self.logger.info(f"✅ Scene graph complete:")
            self.logger.info(f"   📦 Objects: {len(scene_graph.object_nodes)}")
            self.logger.info(f"   🎯 Affordances: {len(scene_graph.affordance_nodes)} ({confident_count} confident, {floating_count} floating)")

            return scene_graph

        except Exception as e:
            self.logger.error(f"Failed to build scene graph: {e}")
            return scene_graph

    def build_all_task_graphs(self, data_root: str, arkitscenes_file: str,
                            visit_id: str, video_id: str) -> List[SceneGraphV2]:
        """Build scene graphs for all tasks in the scene."""
        self.logger.info(f"🏗️ Building scene graphs for all tasks in {visit_id}")

        try:
            # Load task descriptions
            loader = UnifiedDataLoader(data_root, visit_id, video_id)
            scene_data = loader.load_all_data(include_mesh=False, include_point_cloud=False)

            if not scene_data.descriptions:
                self.logger.error("No task descriptions found")
                return []

            # Build graph for each task
            scene_graphs = []
            for task_desc in scene_data.descriptions:
                try:
                    graph = self.build_scene_graph(
                        data_root, arkitscenes_file, visit_id, video_id, task_desc
                    )
                    scene_graphs.append(graph)
                except Exception as e:
                    self.logger.error(f"Failed to build graph for task '{task_desc.get('description', 'unknown')}': {e}")

            self.logger.info(f"✅ Built {len(scene_graphs)} scene graphs")
            return scene_graphs

        except Exception as e:
            self.logger.error(f"Failed to build all task graphs: {e}")
            return []


def save_scene_graph(scene_graph: SceneGraphV2, output_file: str):
    """Save scene graph to JSON file."""
    try:
        graph_dict = scene_graph.to_dict()

        with open(output_file, 'w') as f:
            json.dump(graph_dict, f, indent=2)

        print(f"📝 Scene graph saved to: {output_file}")

    except Exception as e:
        print(f"❌ Failed to save scene graph: {e}")


def main():
    """Test scene graph builder."""
    import argparse

    parser = argparse.ArgumentParser(description="Build scene graphs v2.0")
    parser.add_argument("--data-root", required=True, help="SceneFun3D data root")
    parser.add_argument("--arkitscenes-file", required=True, help="ARKitScenes annotation file")
    parser.add_argument("--visit-id", default="422203", help="Visit ID")
    parser.add_argument("--video-id", default="42445781", help="Video ID")
    parser.add_argument("--output-dir", default="scene_graphs_v2", help="Output directory")
    parser.add_argument("--confidence-threshold", type=float, default=0.3, help="Confidence threshold")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Build scene graphs
    builder = SceneGraphBuilderV2(confidence_threshold=args.confidence_threshold)
    scene_graphs = builder.build_all_task_graphs(
        args.data_root, args.arkitscenes_file, args.visit_id, args.video_id
    )

    # Save graphs
    for i, graph in enumerate(scene_graphs):
        task_name = graph.task_description.lower().replace(' ', '_').replace(',', '')
        output_file = output_dir / f"task_{i+1}_{task_name}_{args.visit_id}.json"
        save_scene_graph(graph, str(output_file))

    print(f"🎉 Built and saved {len(scene_graphs)} scene graphs")


if __name__ == "__main__":
    main()