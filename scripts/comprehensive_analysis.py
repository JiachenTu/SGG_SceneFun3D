"""
Comprehensive Analysis Script

Demonstrates the complete pipeline for analyzing and aligning ARKitScenes and SceneFun3D data.
"""

import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

import numpy as np
from typing import Dict, List

from arkitscenes_parser import ARKitScenesParser
from scenefun3d_parser import SceneFun3DParser
from coordinate_transform import CoordinateTransformer
from point_cloud_utils import PointCloudProcessor
from spatial_analyzer import SpatialAnalyzer
from hierarchical_graph_builder import HierarchicalSceneGraphBuilder


def run_comprehensive_analysis():
    """Run complete analysis pipeline."""
    print("=" * 80)
    print("COMPREHENSIVE ARKITSCENES & SCENEFUN3D ALIGNMENT ANALYSIS")
    print("=" * 80)

    # File paths
    arkitscenes_file = "/home/jiachen/scratch/SceneFun3D/alignment/data_examples/arkitscenes/video_42445781/42445781_3dod_annotation.json"
    scenefun3d_dir = "/home/jiachen/scratch/SceneFun3D/alignment/data_examples/scenefun3d/visit_422203"
    transform_file = "/home/jiachen/scratch/SceneFun3D/alignment/data_examples/scenefun3d/visit_422203/42445781/42445781_transform.npy"
    laser_scan_file = "/home/jiachen/scratch/SceneFun3D/alignment/data_examples/scenefun3d/visit_422203/422203_laser_scan.ply"

    # Step 1: Load and parse data
    print("STEP 1: LOADING AND PARSING DATA")
    print("-" * 40)

    print("Loading ARKitScenes data...")
    arkitscenes_parser = ARKitScenesParser(arkitscenes_file)
    arkitscenes_parser.load()
    arkitscenes_parser.print_summary()

    print("\nLoading SceneFun3D data...")
    scenefun3d_parser = SceneFun3DParser(scenefun3d_dir)
    scenefun3d_parser.load("422203")
    scenefun3d_parser.print_summary()

    print("\nLoading coordinate transformation...")
    transformer = CoordinateTransformer(transform_file)
    transform_info = transformer.get_transformation_info()
    print(f"Transformation angle: {transform_info['rotation_angle_deg']:.2f}°")
    print(f"Translation: {transform_info['translation']}")

    print("\nLoading point cloud...")
    point_processor = PointCloudProcessor(laser_scan_file)
    print(f"Loaded {len(point_processor.points):,} points")

    # Step 2: Spatial analysis
    print("\n" + "=" * 80)
    print("STEP 2: SPATIAL ANALYSIS")
    print("-" * 40)

    analyzer = SpatialAnalyzer(arkitscenes_parser, scenefun3d_parser,
                               transformer, point_processor)

    relationships = analyzer.analyze_all_spatial_relationships()
    analyzer.print_spatial_analysis_summary(relationships)

    # Step 3: Build hierarchical scene graphs
    print("\n" + "=" * 80)
    print("STEP 3: HIERARCHICAL SCENE GRAPH CONSTRUCTION")
    print("-" * 40)

    builder = HierarchicalSceneGraphBuilder(
        arkitscenes_parser, scenefun3d_parser, transformer, point_processor
    )

    # Build scene graphs for all tasks
    all_graphs = builder.build_all_scene_graphs()
    print(f"Built {len(all_graphs)} hierarchical scene graphs")

    # Step 4: Detailed analysis of specific tasks
    print("\n" + "=" * 80)
    print("STEP 4: DETAILED TASK ANALYSIS")
    print("-" * 40)

    # Analyze flush toilet task
    flush_task = scenefun3d_parser.get_task_by_description("flush")
    if flush_task:
        print("FLUSH TOILET TASK ANALYSIS:")
        print("-" * 30)

        flush_graph = builder.build_scene_graph_for_task(flush_task)
        builder.print_scene_graph_summary(flush_graph)

        # Save detailed scene graph
        output_file = "/home/jiachen/scratch/SceneFun3D/alignment/flush_toilet_scene_graph.json"
        builder.save_scene_graph(flush_graph, output_file)
        print(f"Detailed scene graph saved to: {output_file}")

        # Analyze the flush affordance specifically
        analyze_flush_affordance(flush_task, relationships, point_processor, transformer)

    # Analyze tap task
    tap_task = scenefun3d_parser.get_task_by_description("tap")
    if tap_task:
        print("\nTAP TASK ANALYSIS:")
        print("-" * 20)

        tap_graph = builder.build_scene_graph_for_task(tap_task)
        builder.print_scene_graph_summary(tap_graph)

        # Save tap scene graph
        output_file = "/home/jiachen/scratch/SceneFun3D/alignment/tap_scene_graph.json"
        builder.save_scene_graph(tap_graph, output_file)
        print(f"Tap scene graph saved to: {output_file}")

    # Step 5: Summary statistics
    print("\n" + "=" * 80)
    print("STEP 5: SUMMARY STATISTICS")
    print("-" * 40)

    print_comprehensive_statistics(arkitscenes_parser, scenefun3d_parser,
                                   relationships, all_graphs)

    # Step 6: Export results
    print("\n" + "=" * 80)
    print("STEP 6: EXPORTING RESULTS")
    print("-" * 40)

    export_all_results(all_graphs, relationships)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)


def analyze_flush_affordance(task, relationships, point_processor, transformer):
    """Detailed analysis of the flush toilet affordance."""
    print("\nDETAILED FLUSH AFFORDANCE ANALYSIS:")

    if not task.annotations:
        print("No annotations found for flush task")
        return

    annotation = task.annotations[0]  # Primary annotation
    print(f"Annotation ID: {annotation.annot_id}")
    print(f"Point count: {len(annotation.indices)}")

    # Get motion parameters
    if annotation.motion:
        motion = annotation.motion
        print(f"Motion type: {motion.motion_type}")
        print(f"Motion direction: {motion.motion_dir}")
        print(f"Motion origin index: {motion.motion_origin_idx}")

        # Convert motion origin to 3D coordinates
        if 0 <= motion.motion_origin_idx < len(point_processor.points):
            origin_3d = point_processor.points[motion.motion_origin_idx]
            transformed_origin = transformer.transform_points(origin_3d.reshape(1, -1))
            print(f"Motion origin (3D): {transformed_origin[0]}")

    # Get affordance bounding box
    affordance_points = point_processor.get_annotation_points(annotation)
    transformed_points = transformer.transform_points(affordance_points)

    if len(transformed_points) > 0:
        min_bounds = np.min(transformed_points, axis=0)
        max_bounds = np.max(transformed_points, axis=0)
        center = (min_bounds + max_bounds) / 2
        size = max_bounds - min_bounds

        print(f"Affordance center: {center}")
        print(f"Affordance size: {size}")

    # Find spatial relationship
    for rel in relationships:
        if rel.annotation_id == annotation.annot_id:
            print(f"Parent object: {rel.object_label}")
            print(f"Overlap ratio: {rel.overlap_ratio:.3f}")
            print(f"Distance to object: {rel.distance:.1f} mm")
            print(f"Confidence: {rel.confidence:.3f}")
            break


def print_comprehensive_statistics(arkitscenes_parser, scenefun3d_parser,
                                   relationships, scene_graphs):
    """Print comprehensive statistics."""
    print("COMPREHENSIVE STATISTICS:")

    # ARKitScenes stats
    objects = arkitscenes_parser.get_objects()
    print(f"ARKitScenes objects: {len(objects)}")
    for obj in objects:
        volume = obj.get_volume() / 1e9  # Convert to m³
        print(f"  {obj.label}: {volume:.3f} m³")

    # SceneFun3D stats
    print(f"\nSceneFun3D tasks: {len(scenefun3d_parser.get_task_descriptions())}")
    print(f"Total annotations: {len(scenefun3d_parser.annotations)}")

    total_points = sum(len(ann.indices) for ann in scenefun3d_parser.annotations.values())
    print(f"Total annotated points: {total_points:,}")

    # Spatial relationship stats
    print(f"\nSpatial relationships: {len(relationships)}")
    high_conf = sum(1 for rel in relationships if rel.confidence > 0.7)
    print(f"High confidence relationships: {high_conf}")

    # Scene graph stats
    total_nodes = sum(len(graph.nodes) for graph in scene_graphs)
    print(f"\nScene graph nodes: {total_nodes}")
    print(f"Average nodes per task: {total_nodes / len(scene_graphs):.1f}")


def export_all_results(scene_graphs, relationships):
    """Export all results to files."""
    output_dir = "/home/jiachen/scratch/SceneFun3D/alignment"

    # Export all scene graphs
    for i, graph in enumerate(scene_graphs):
        filename = f"scene_graph_{i+1}_{graph.task_id[:8]}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(graph.to_dict(), f, indent=2)
        print(f"Exported: {filename}")

    # Export spatial relationships summary
    relationships_summary = []
    for rel in relationships:
        relationships_summary.append({
            'object_label': rel.object_label,
            'affordance_type': rel.affordance_type,
            'task_description': rel.task_description,
            'overlap_ratio': rel.overlap_ratio,
            'distance': rel.distance,
            'confidence': rel.confidence,
            'motion_type': rel.motion_type
        })

    relationships_file = os.path.join(output_dir, "spatial_relationships_summary.json")
    with open(relationships_file, 'w') as f:
        json.dump(relationships_summary, f, indent=2)
    print(f"Exported: spatial_relationships_summary.json")

    # Export analysis summary
    summary = {
        'dataset_info': {
            'visit_id': '422203',
            'video_id': '42445781',
            'scene_type': 'bathroom'
        },
        'statistics': {
            'num_objects': len(scene_graphs[0].get_nodes_by_type('object')) if scene_graphs else 0,
            'num_tasks': len(scene_graphs),
            'num_relationships': len(relationships),
            'high_confidence_relationships': sum(1 for rel in relationships if rel.confidence > 0.7)
        },
        'files_generated': [
            'scene_graph_*.json',
            'spatial_relationships_summary.json',
            'analysis_summary.json'
        ]
    }

    summary_file = os.path.join(output_dir, "analysis_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Exported: analysis_summary.json")


if __name__ == "__main__":
    run_comprehensive_analysis()