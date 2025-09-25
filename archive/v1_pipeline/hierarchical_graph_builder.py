"""
Hierarchical Scene Graph Builder

Builds hierarchical 3D scene graphs combining ARKitScenes objects and SceneFun3D affordances.
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
    from scripts.spatial_analyzer import SpatialAnalyzer, SpatialRelationship
except ImportError:
    # Fall back to path-based imports (when run standalone)
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
    sys.path.append(os.path.dirname(__file__))
    from arkitscenes_parser import ARKitScenesParser, ARKitObject
    from scenefun3d_parser import SceneFun3DParser, TaskDescription, Annotation
    from coordinate_transform import CoordinateTransformer
    from point_cloud_utils import PointCloudProcessor
    from spatial_analyzer import SpatialAnalyzer, SpatialRelationship

import json
import uuid
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field



@dataclass
class SceneGraphNode:
    """Base class for scene graph nodes."""
    node_id: str
    node_type: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)


@dataclass
class SceneRootNode(SceneGraphNode):
    """Root node representing the entire scene."""
    scene_description: str = field(default="")
    visit_id: str = field(default="")
    video_id: str = field(default="")
    spatial_bounds: Dict[str, List[float]] = field(default_factory=dict)

    def __post_init__(self):
        self.node_type = "scene_root"


@dataclass
class SpatialRegionNode(SceneGraphNode):
    """Node representing a spatial region (e.g., toilet area, sink area)."""
    region_name: str = field(default="")
    region_bounds: Dict[str, List[float]] = field(default_factory=dict)
    primary_objects: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.node_type = "spatial_region"


@dataclass
class ObjectNode(SceneGraphNode):
    """Node representing a physical object."""
    semantic_class: str = field(default="")
    object_uid: str = field(default="")
    bbox_center: List[float] = field(default_factory=list)
    bbox_size: List[float] = field(default_factory=list)
    bbox_rotation: List[float] = field(default_factory=list)  # Flattened 3x3 rotation matrix
    volume: float = field(default=0.0)
    attributes: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.node_type = "object"


@dataclass
class AffordanceNode(SceneGraphNode):
    """Node representing an affordance."""
    affordance_type: str = field(default="")
    annotation_id: str = field(default="")
    motion_type: Optional[str] = field(default=None)
    motion_direction: Optional[List[float]] = field(default=None)
    motion_origin: Optional[List[float]] = field(default=None)
    confidence: float = field(default=0.0)
    point_count: int = field(default=0)
    bbox_center: List[float] = field(default_factory=list)
    bbox_size: List[float] = field(default_factory=list)
    attached_object_id: str = field(default="")

    def __post_init__(self):
        self.node_type = "affordance"


@dataclass
class TaskSceneGraph:
    """Complete scene graph for a specific task."""
    task_description: str = field(default="")
    task_id: str = field(default="")
    visit_id: str = field(default="")
    video_id: str = field(default="")
    nodes: Dict[str, SceneGraphNode] = field(default_factory=dict)
    root_id: str = field(default="")
    target_affordances: List[str] = field(default_factory=list)  # IDs of affordances relevant to the task
    spatial_reasoning_chain: List[str] = field(default_factory=list)  # Step-by-step reasoning

    def get_node(self, node_id: str) -> Optional[SceneGraphNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def get_nodes_by_type(self, node_type: str) -> List[SceneGraphNode]:
        """Get all nodes of a specific type."""
        return [node for node in self.nodes.values() if node.node_type == node_type]

    def get_children(self, node_id: str) -> List[SceneGraphNode]:
        """Get children of a node."""
        node = self.get_node(node_id)
        if not node:
            return []
        return [self.nodes[child_id] for child_id in node.children_ids if child_id in self.nodes]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'task_description': self.task_description,
            'task_id': self.task_id,
            'visit_id': self.visit_id,
            'video_id': self.video_id,
            'root_id': self.root_id,
            'target_affordances': self.target_affordances,
            'spatial_reasoning_chain': self.spatial_reasoning_chain,
            'nodes': {
                node_id: asdict(node) for node_id, node in self.nodes.items()
            }
        }


class HierarchicalSceneGraphBuilder:
    """Builds hierarchical scene graphs from aligned data."""

    def __init__(self, arkitscenes_parser: ARKitScenesParser,
                 scenefun3d_parser: SceneFun3DParser,
                 transformer: CoordinateTransformer,
                 point_processor: PointCloudProcessor):
        """Initialize graph builder."""
        self.arkitscenes_parser = arkitscenes_parser
        self.scenefun3d_parser = scenefun3d_parser
        self.transformer = transformer
        self.point_processor = point_processor

        # Initialize spatial analyzer
        self.spatial_analyzer = SpatialAnalyzer(
            arkitscenes_parser, scenefun3d_parser, transformer, point_processor
        )

    def generate_node_id(self, prefix: str = "node") -> str:
        """Generate unique node ID."""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def create_scene_root(self, visit_id: str, video_id: str) -> SceneRootNode:
        """Create scene root node."""
        # Compute scene bounds from all objects
        objects = self.arkitscenes_parser.get_objects()
        if objects:
            all_corners = []
            for obj in objects:
                corners = obj.obb.get_corners()
                all_corners.append(corners)
            all_corners = np.vstack(all_corners)
            min_bounds = np.min(all_corners, axis=0).tolist()
            max_bounds = np.max(all_corners, axis=0).tolist()
        else:
            min_bounds = [0, 0, 0]
            max_bounds = [0, 0, 0]

        # Infer scene description from objects
        object_labels = [obj.label for obj in objects]
        if 'toilet' in object_labels and 'sink' in object_labels:
            scene_desc = "Bathroom scene"
        else:
            scene_desc = f"Indoor scene with {', '.join(set(object_labels))}"

        return SceneRootNode(
            node_id=self.generate_node_id("scene"),
            node_type="scene_root",
            scene_description=scene_desc,
            visit_id=visit_id,
            video_id=video_id,
            spatial_bounds={
                'min': min_bounds,
                'max': max_bounds,
                'size': (np.array(max_bounds) - np.array(min_bounds)).tolist()
            }
        )

    def create_spatial_regions(self, objects: List[ARKitObject]) -> List[SpatialRegionNode]:
        """Create spatial region nodes based on object groupings."""
        regions = []

        # Group objects by semantic similarity and spatial proximity
        object_groups = self._group_objects_into_regions(objects)

        for region_name, region_objects in object_groups.items():
            # Compute region bounds
            all_corners = []
            for obj in region_objects:
                corners = obj.obb.get_corners()
                all_corners.append(corners)

            if all_corners:
                all_corners = np.vstack(all_corners)
                min_bounds = np.min(all_corners, axis=0).tolist()
                max_bounds = np.max(all_corners, axis=0).tolist()
            else:
                min_bounds = [0, 0, 0]
                max_bounds = [0, 0, 0]

            region = SpatialRegionNode(
                node_id=self.generate_node_id("region"),
                node_type="spatial_region",
                region_name=region_name,
                region_bounds={
                    'min': min_bounds,
                    'max': max_bounds,
                    'size': (np.array(max_bounds) - np.array(min_bounds)).tolist()
                },
                primary_objects=[obj.label for obj in region_objects]
            )
            regions.append(region)

        return regions

    def _group_objects_into_regions(self, objects: List[ARKitObject]) -> Dict[str, List[ARKitObject]]:
        """Group objects into spatial regions."""
        groups = {}

        for obj in objects:
            # Simple grouping based on object type
            if obj.label in ['toilet']:
                region_name = "toilet_area"
            elif obj.label in ['sink']:
                region_name = "sink_area"
            elif obj.label in ['bathtub']:
                region_name = "bathtub_area"
            else:
                region_name = "general_area"

            if region_name not in groups:
                groups[region_name] = []
            groups[region_name].append(obj)

        return groups

    def create_object_nodes(self, objects: List[ARKitObject]) -> List[ObjectNode]:
        """Create object nodes from ARKitScenes objects."""
        object_nodes = []

        for obj in objects:
            node = ObjectNode(
                node_id=self.generate_node_id("obj"),
                node_type="object",
                semantic_class=obj.label,
                object_uid=obj.uid,
                bbox_center=obj.obb.centroid.tolist(),
                bbox_size=obj.obb.axes_lengths.tolist(),
                bbox_rotation=obj.obb.normalized_axes.tolist(),
                volume=obj.get_volume(),
                attributes=obj.attributes
            )
            object_nodes.append(node)

        return object_nodes

    def create_affordance_nodes(self, task: TaskDescription,
                                relationships: List[SpatialRelationship]) -> List[AffordanceNode]:
        """Create affordance nodes for a task."""
        affordance_nodes = []

        for annotation in task.annotations:
            # Find spatial relationship for this annotation
            relationship = None
            for rel in relationships:
                if rel.annotation_id == annotation.annot_id:
                    relationship = rel
                    break

            if not relationship:
                continue

            # Get affordance points and compute bounding box
            affordance_points = self.point_processor.get_annotation_points(annotation)
            transformed_points = self.transformer.transform_points(affordance_points)

            if len(transformed_points) > 0:
                min_bounds = np.min(transformed_points, axis=0)
                max_bounds = np.max(transformed_points, axis=0)
                center = (min_bounds + max_bounds) / 2
                size = max_bounds - min_bounds
            else:
                center = np.zeros(3)
                size = np.zeros(3)

            # Extract motion parameters
            motion_direction = None
            motion_origin = None
            motion_type = None

            if annotation.motion:
                motion_type = annotation.motion.motion_type
                motion_direction = annotation.motion.motion_dir.tolist()

                # Convert motion origin index to 3D coordinate
                origin_idx = annotation.motion.motion_origin_idx
                if 0 <= origin_idx < len(self.point_processor.points):
                    origin_point = self.point_processor.points[origin_idx:origin_idx+1]
                    transformed_origin = self.transformer.transform_points(origin_point)
                    motion_origin = transformed_origin[0].tolist()

            node = AffordanceNode(
                node_id=self.generate_node_id("aff"),
                node_type="affordance",
                affordance_type=relationship.affordance_type,
                annotation_id=annotation.annot_id,
                motion_type=motion_type,
                motion_direction=motion_direction,
                motion_origin=motion_origin,
                confidence=relationship.confidence,
                point_count=len(annotation.indices),
                bbox_center=center.tolist(),
                bbox_size=size.tolist(),
                attached_object_id=""  # Will be set later
            )

            affordance_nodes.append(node)

        return affordance_nodes

    def build_scene_graph_for_task(self, task: TaskDescription) -> TaskSceneGraph:
        """Build complete scene graph for a specific task."""
        print(f"Building scene graph for task: \"{task.description}\"")

        # Analyze spatial relationships for this task
        relationships = self.spatial_analyzer.analyze_task_spatial_relationships(task)

        # Create nodes
        objects = self.arkitscenes_parser.get_objects()

        # Level 0: Scene root
        scene_root = self.create_scene_root(
            self.scenefun3d_parser.visit_id,
            "42445781"  # TODO: Make this dynamic
        )

        # Level 1: Spatial regions
        spatial_regions = self.create_spatial_regions(objects)

        # Level 2: Object nodes
        object_nodes = self.create_object_nodes(objects)

        # Level 3: Affordance nodes
        affordance_nodes = self.create_affordance_nodes(task, relationships)

        # Build hierarchy
        all_nodes = {}

        # Add scene root
        all_nodes[scene_root.node_id] = scene_root

        # Add spatial regions as children of scene root
        scene_root.children_ids = [region.node_id for region in spatial_regions]
        for region in spatial_regions:
            region.parent_id = scene_root.node_id
            all_nodes[region.node_id] = region

        # Add objects to appropriate regions
        self._assign_objects_to_regions(object_nodes, spatial_regions, objects)
        for obj_node in object_nodes:
            all_nodes[obj_node.node_id] = obj_node

        # Add affordances to objects
        self._assign_affordances_to_objects(affordance_nodes, object_nodes, relationships)
        for aff_node in affordance_nodes:
            all_nodes[aff_node.node_id] = aff_node

        # Determine target affordances and reasoning chain
        target_affordances = [node.node_id for node in affordance_nodes]
        reasoning_chain = self._generate_spatial_reasoning_chain(task, relationships)

        # Create final scene graph
        scene_graph = TaskSceneGraph(
            task_description=task.description,
            task_id=task.desc_id,
            visit_id=self.scenefun3d_parser.visit_id,
            video_id="42445781",
            nodes=all_nodes,
            root_id=scene_root.node_id,
            target_affordances=target_affordances,
            spatial_reasoning_chain=reasoning_chain
        )

        return scene_graph

    def _assign_objects_to_regions(self, object_nodes: List[ObjectNode],
                                   spatial_regions: List[SpatialRegionNode],
                                   arkit_objects: List[ARKitObject]) -> None:
        """Assign object nodes to spatial regions."""
        # Create mapping from ARKit object to object node
        arkit_to_node = {}
        for obj_node, arkit_obj in zip(object_nodes, arkit_objects):
            arkit_to_node[arkit_obj.uid] = obj_node

        # Group objects and assign to regions
        object_groups = self._group_objects_into_regions(arkit_objects)

        for region in spatial_regions:
            region_objects = object_groups.get(region.region_name, [])
            for arkit_obj in region_objects:
                if arkit_obj.uid in arkit_to_node:
                    obj_node = arkit_to_node[arkit_obj.uid]
                    obj_node.parent_id = region.node_id
                    region.children_ids.append(obj_node.node_id)

    def _assign_affordances_to_objects(self, affordance_nodes: List[AffordanceNode],
                                       object_nodes: List[ObjectNode],
                                       relationships: List[SpatialRelationship]) -> None:
        """Assign affordance nodes to object nodes."""
        # Create mapping from object UID to object node
        uid_to_node = {node.object_uid: node for node in object_nodes}

        for aff_node in affordance_nodes:
            # Find corresponding relationship
            relationship = None
            for rel in relationships:
                if rel.annotation_id == aff_node.annotation_id:
                    relationship = rel
                    break

            if relationship and relationship.object_id in uid_to_node:
                parent_obj_node = uid_to_node[relationship.object_id]
                aff_node.parent_id = parent_obj_node.node_id
                aff_node.attached_object_id = parent_obj_node.node_id
                parent_obj_node.children_ids.append(aff_node.node_id)

    def _generate_spatial_reasoning_chain(self, task: TaskDescription,
                                          relationships: List[SpatialRelationship]) -> List[str]:
        """Generate spatial reasoning chain for the task."""
        if not relationships:
            return ["No spatial relationships found"]

        rel = relationships[0]  # Primary relationship

        steps = [
            f"Locate {rel.object_label} in the scene",
            f"Identify {rel.affordance_type.lower()} affordance on {rel.object_label}",
            f"Execute {rel.motion_type} motion" if rel.motion_type else f"Interact with {rel.affordance_type.lower()}"
        ]

        return steps

    def build_all_scene_graphs(self) -> List[TaskSceneGraph]:
        """Build scene graphs for all tasks."""
        scene_graphs = []

        for task in self.scenefun3d_parser.get_task_descriptions():
            graph = self.build_scene_graph_for_task(task)
            scene_graphs.append(graph)

        return scene_graphs

    def save_scene_graph(self, scene_graph: TaskSceneGraph, output_file: str) -> None:
        """Save scene graph to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(scene_graph.to_dict(), f, indent=2)

    def print_scene_graph_summary(self, scene_graph: TaskSceneGraph) -> None:
        """Print summary of a scene graph."""
        print(f"\nScene Graph Summary:")
        print(f"Task: \"{scene_graph.task_description}\"")
        print(f"Total nodes: {len(scene_graph.nodes)}")

        # Count nodes by type
        node_types = {}
        for node in scene_graph.nodes.values():
            node_type = node.node_type
            node_types[node_type] = node_types.get(node_type, 0) + 1

        for node_type, count in node_types.items():
            print(f"  {node_type}: {count}")

        print(f"Target affordances: {len(scene_graph.target_affordances)}")
        print(f"Reasoning steps: {len(scene_graph.spatial_reasoning_chain)}")

        for i, step in enumerate(scene_graph.spatial_reasoning_chain, 1):
            print(f"  {i}. {step}")


def main():
    """Example usage of hierarchical scene graph builder."""
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

    # Create graph builder
    builder = HierarchicalSceneGraphBuilder(
        arkitscenes_parser, scenefun3d_parser, transformer, point_processor
    )

    print("Building scene graphs...")

    # Build scene graph for "Flush the toilet" task
    flush_task = scenefun3d_parser.get_task_by_description("flush")
    if flush_task:
        flush_graph = builder.build_scene_graph_for_task(flush_task)
        builder.print_scene_graph_summary(flush_graph)

        # Save scene graph
        output_file = "/home/jiachen/scratch/SceneFun3D/alignment/flush_toilet_scene_graph.json"
        builder.save_scene_graph(flush_graph, output_file)
        print(f"\nScene graph saved to: {output_file}")

    # Build all scene graphs
    print("\n" + "=" * 60)
    print("BUILDING ALL SCENE GRAPHS")
    print("=" * 60)

    all_graphs = builder.build_all_scene_graphs()
    print(f"Built {len(all_graphs)} scene graphs")

    for graph in all_graphs:
        print(f"\n\"{graph.task_description}\" - {len(graph.nodes)} nodes")


if __name__ == "__main__":
    main()