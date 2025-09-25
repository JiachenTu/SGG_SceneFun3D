#!/usr/bin/env python3
"""
Integrated Scene Graph Generation Pipeline

This script runs the complete pipeline to generate hierarchical scene graphs
from ARKitScenes and SceneFun3D data.

Usage:
    python run_pipeline.py [--validate] [--verbose] [--output-dir OUTPUT_DIR]
"""

import sys
import os
import json
import argparse
import traceback
from datetime import datetime
from pathlib import Path

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

import numpy as np

try:
    from arkitscenes_parser import ARKitScenesParser
    from scenefun3d_parser import SceneFun3DParser
    from coordinate_transform import CoordinateTransformer
    from point_cloud_utils import PointCloudProcessor
except ImportError as e:
    print(f"Error importing utilities: {e}")
    print("Make sure you're in the correct directory and have the required dependencies installed.")
    sys.exit(1)

# Import analysis modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

try:
    from spatial_analyzer import SpatialAnalyzer
    from hierarchical_graph_builder import HierarchicalSceneGraphBuilder
    from scene_graph_visualizer import generate_pipeline_visualizations
except ImportError as e:
    print(f"Error importing analysis modules: {e}")
    print("Make sure all scripts are available in the scripts/ directory.")
    sys.exit(1)


class SceneGraphPipeline:
    """Integrated pipeline for scene graph generation."""

    def __init__(self, output_dir: str = "outputs", verbose: bool = False):
        """Initialize pipeline."""
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.execution_log = []

        # Create output directories
        self.create_output_directories()

        # Pipeline components
        self.arkitscenes_parser = None
        self.scenefun3d_parser = None
        self.transformer = None
        self.point_processor = None
        self.spatial_analyzer = None
        self.graph_builder = None

    def create_output_directories(self):
        """Create output directory structure."""
        directories = [
            self.output_dir,
            self.output_dir / "scene_graphs",
            self.output_dir / "spatial_analysis",
            self.output_dir / "validation",
            self.output_dir / "visualizations",
            self.output_dir / "logs"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        self.log(f"Created output directories in: {self.output_dir}")

    def log(self, message: str, level: str = "INFO"):
        """Log a message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"

        if self.verbose:
            print(log_entry)

        self.execution_log.append(log_entry)

    def validate_input_files(self) -> bool:
        """Validate that all required input files exist."""
        base_dir = Path(__file__).parent / "data_examples"

        required_files = [
            base_dir / "arkitscenes" / "video_42445781" / "42445781_3dod_annotation.json",
            base_dir / "scenefun3d" / "visit_422203" / "422203_descriptions.json",
            base_dir / "scenefun3d" / "visit_422203" / "422203_annotations.json",
            base_dir / "scenefun3d" / "visit_422203" / "422203_motions.json",
            base_dir / "scenefun3d" / "visit_422203" / "422203_laser_scan.ply",
            base_dir / "scenefun3d" / "visit_422203" / "42445781" / "42445781_transform.npy"
        ]

        missing_files = []
        for file_path in required_files:
            if not file_path.exists():
                missing_files.append(str(file_path))

        if missing_files:
            self.log(f"Missing required files: {missing_files}", "ERROR")
            return False

        self.log("All required input files found")
        return True

    def load_data(self) -> bool:
        """Load and parse all data sources."""
        try:
            base_dir = Path(__file__).parent / "data_examples"

            # File paths
            arkitscenes_file = base_dir / "arkitscenes" / "video_42445781" / "42445781_3dod_annotation.json"
            scenefun3d_dir = base_dir / "scenefun3d" / "visit_422203"
            transform_file = scenefun3d_dir / "42445781" / "42445781_transform.npy"
            laser_scan_file = scenefun3d_dir / "422203_laser_scan.ply"

            self.log("Loading ARKitScenes data...")
            self.arkitscenes_parser = ARKitScenesParser(str(arkitscenes_file))
            self.arkitscenes_parser.load()
            objects = self.arkitscenes_parser.get_objects()
            self.log(f"Loaded {len(objects)} ARKitScenes objects: {[obj.label for obj in objects]}")

            self.log("Loading SceneFun3D data...")
            self.scenefun3d_parser = SceneFun3DParser(str(scenefun3d_dir))
            self.scenefun3d_parser.load("422203")
            tasks = self.scenefun3d_parser.get_task_descriptions()
            self.log(f"Loaded {len(tasks)} SceneFun3D tasks")

            self.log("Loading coordinate transformation...")
            self.transformer = CoordinateTransformer(str(transform_file))
            transform_info = self.transformer.get_transformation_info()
            self.log(f"Transform loaded - rotation: {transform_info['rotation_angle_deg']:.2f}°")

            self.log("Loading point cloud...")
            self.point_processor = PointCloudProcessor(str(laser_scan_file))
            self.log(f"Loaded point cloud with {len(self.point_processor.points):,} points")

            return True

        except Exception as e:
            self.log(f"Error loading data: {e}", "ERROR")
            if self.verbose:
                traceback.print_exc()
            return False

    def run_spatial_analysis(self) -> bool:
        """Run spatial analysis to find object-affordance relationships."""
        try:
            self.log("Running spatial analysis...")

            self.spatial_analyzer = SpatialAnalyzer(
                self.arkitscenes_parser,
                self.scenefun3d_parser,
                self.transformer,
                self.point_processor
            )

            relationships = self.spatial_analyzer.analyze_all_spatial_relationships()
            self.log(f"Found {len(relationships)} spatial relationships")

            # Save spatial analysis results
            relationships_data = []
            for rel in relationships:
                relationships_data.append({
                    'object_label': rel.object_label,
                    'object_id': rel.object_id,
                    'annotation_id': rel.annotation_id,
                    'task_description': rel.task_description,
                    'affordance_type': rel.affordance_type,
                    'overlap_ratio': rel.overlap_ratio,
                    'distance': rel.distance,
                    'confidence': rel.confidence,
                    'motion_type': rel.motion_type
                })

            output_file = self.output_dir / "spatial_analysis" / "spatial_relationships.json"
            with open(output_file, 'w') as f:
                json.dump(relationships_data, f, indent=2)

            self.log(f"Spatial analysis results saved to: {output_file}")

            # Store for later use
            self.spatial_relationships = relationships

            return True

        except Exception as e:
            self.log(f"Error in spatial analysis: {e}", "ERROR")
            if self.verbose:
                traceback.print_exc()
            return False

    def build_scene_graphs(self) -> bool:
        """Build hierarchical scene graphs for all tasks."""
        try:
            self.log("Building hierarchical scene graphs...")

            self.graph_builder = HierarchicalSceneGraphBuilder(
                self.arkitscenes_parser,
                self.scenefun3d_parser,
                self.transformer,
                self.point_processor
            )

            tasks = self.scenefun3d_parser.get_task_descriptions()
            scene_graphs = []

            for i, task in enumerate(tasks):
                self.log(f"Building scene graph {i+1}/{len(tasks)}: \"{task.description}\"")

                try:
                    scene_graph = self.graph_builder.build_scene_graph_for_task(task)
                    scene_graphs.append(scene_graph)

                    # Generate clean filename
                    clean_desc = task.description.lower()
                    clean_desc = ''.join(c if c.isalnum() or c == ' ' else '' for c in clean_desc)
                    clean_desc = '_'.join(clean_desc.split())

                    filename = f"bathroom_422203_task_{i+1}_{clean_desc}.json"
                    output_file = self.output_dir / "scene_graphs" / filename

                    # Save scene graph
                    with open(output_file, 'w') as f:
                        json.dump(scene_graph.to_dict(), f, indent=2)

                    self.log(f"Scene graph saved: {filename}")

                except Exception as e:
                    self.log(f"Error building scene graph for task '{task.description}': {e}", "ERROR")
                    continue

            self.log(f"Successfully built {len(scene_graphs)} scene graphs")
            self.scene_graphs = scene_graphs

            return len(scene_graphs) > 0

        except Exception as e:
            self.log(f"Error building scene graphs: {e}", "ERROR")
            if self.verbose:
                traceback.print_exc()
            return False

    def generate_visualizations(self) -> bool:
        """Generate scene graph visualizations."""
        try:
            self.log("Generating scene graph visualizations...")

            scene_graphs_dir = str(self.output_dir / "scene_graphs")
            output_dir = str(self.output_dir / "visualizations")

            success = generate_pipeline_visualizations(
                scene_graphs_dir=scene_graphs_dir,
                output_dir=output_dir,
                verbose=self.verbose
            )

            if success:
                self.log("Scene graph visualizations generated successfully")
                # Count generated files
                viz_dir = self.output_dir / "visualizations"
                if viz_dir.exists():
                    png_files = list(viz_dir.glob("*.png"))
                    svg_files = list(viz_dir.glob("*.svg"))
                    self.log(f"Generated {len(png_files)} PNG and {len(svg_files)} SVG visualizations")
            else:
                self.log("Failed to generate visualizations", "ERROR")

            return success

        except Exception as e:
            self.log(f"Error generating visualizations: {e}", "ERROR")
            if self.verbose:
                traceback.print_exc()
            return False

    def generate_summary_report(self) -> bool:
        """Generate comprehensive summary report."""
        try:
            self.log("Generating summary report...")

            # Collect statistics
            objects = self.arkitscenes_parser.get_objects()
            tasks = self.scenefun3d_parser.get_task_descriptions()

            summary = {
                'execution_timestamp': datetime.now().isoformat(),
                'data_info': {
                    'visit_id': '422203',
                    'video_id': '42445781',
                    'scene_type': 'bathroom'
                },
                'input_statistics': {
                    'arkitscenes_objects': len(objects),
                    'object_labels': [obj.label for obj in objects],
                    'scenefun3d_tasks': len(tasks),
                    'task_descriptions': [task.description for task in tasks],
                    'total_annotations': len(self.scenefun3d_parser.annotations),
                    'total_motions': len(self.scenefun3d_parser.motions)
                },
                'processing_results': {
                    'spatial_relationships_found': len(self.spatial_relationships),
                    'scene_graphs_generated': len(self.scene_graphs),
                    'high_confidence_relationships': sum(
                        1 for rel in self.spatial_relationships if rel.confidence > 0.7
                    )
                },
                'output_files': {
                    'scene_graphs': [f.name for f in (self.output_dir / "scene_graphs").glob("*.json")],
                    'spatial_analysis': [f.name for f in (self.output_dir / "spatial_analysis").glob("*.json")],
                    'visualizations': [f.name for f in (self.output_dir / "visualizations").glob("*.*")]
                }
            }

            # Add detailed relationship analysis
            relationships_by_object = {}
            for rel in self.spatial_relationships:
                if rel.object_label not in relationships_by_object:
                    relationships_by_object[rel.object_label] = []
                relationships_by_object[rel.object_label].append({
                    'task': rel.task_description,
                    'affordance': rel.affordance_type,
                    'confidence': rel.confidence
                })

            summary['relationship_analysis'] = relationships_by_object

            # Save summary
            summary_file = self.output_dir / "pipeline_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            self.log(f"Summary report saved to: {summary_file}")
            return True

        except Exception as e:
            self.log(f"Error generating summary report: {e}", "ERROR")
            if self.verbose:
                traceback.print_exc()
            return False

    def save_execution_log(self):
        """Save execution log to file."""
        log_file = self.output_dir / "logs" / f"pipeline_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        with open(log_file, 'w') as f:
            for entry in self.execution_log:
                f.write(entry + '\n')

        if self.verbose:
            print(f"Execution log saved to: {log_file}")

    def run_pipeline(self, validate: bool = False) -> bool:
        """Run the complete pipeline."""
        self.log("=" * 80)
        self.log("STARTING SCENE GRAPH GENERATION PIPELINE")
        self.log("=" * 80)

        success = True

        # Step 1: Validate inputs
        if not self.validate_input_files():
            self.log("Input validation failed", "ERROR")
            return False

        # Step 2: Load data
        if not self.load_data():
            self.log("Data loading failed", "ERROR")
            return False

        # Step 3: Spatial analysis
        if not self.run_spatial_analysis():
            self.log("Spatial analysis failed", "ERROR")
            success = False

        # Step 4: Build scene graphs
        if success and not self.build_scene_graphs():
            self.log("Scene graph construction failed", "ERROR")
            success = False

        # Step 5: Generate visualizations
        if success and not self.generate_visualizations():
            self.log("Visualization generation failed", "ERROR")
            success = False

        # Step 6: Generate summary
        if success and not self.generate_summary_report():
            self.log("Summary generation failed", "ERROR")
            success = False

        # Step 7: Run validation if requested
        if validate and success:
            success = self.run_validation()

        # Save log
        self.save_execution_log()

        if success:
            self.log("=" * 80)
            self.log("PIPELINE COMPLETED SUCCESSFULLY")
            self.log(f"Results saved to: {self.output_dir}")
            self.log("=" * 80)
        else:
            self.log("=" * 80)
            self.log("PIPELINE COMPLETED WITH ERRORS")
            self.log("=" * 80)

        return success

    def run_validation(self) -> bool:
        """Run validation checks on outputs."""
        self.log("Running validation checks...")

        try:
            validation_results = {
                'timestamp': datetime.now().isoformat(),
                'checks': []
            }

            # Check 1: Scene graph completeness
            scene_graph_files = list((self.output_dir / "scene_graphs").glob("*.json"))
            expected_count = len(self.scenefun3d_parser.get_task_descriptions())

            validation_results['checks'].append({
                'check': 'scene_graph_count',
                'expected': expected_count,
                'actual': len(scene_graph_files),
                'passed': len(scene_graph_files) == expected_count
            })

            # Check 2: Spatial relationships quality
            high_conf_count = sum(1 for rel in self.spatial_relationships if rel.confidence > 0.7)
            total_relationships = len(self.spatial_relationships)

            validation_results['checks'].append({
                'check': 'high_confidence_relationships',
                'high_confidence_count': high_conf_count,
                'total_relationships': total_relationships,
                'high_confidence_ratio': high_conf_count / total_relationships if total_relationships > 0 else 0,
                'passed': high_conf_count > 0
            })

            # Check 3: Scene graph structure
            valid_graphs = 0
            for graph in self.scene_graphs:
                if (len(graph.nodes) > 0 and
                    len(graph.target_affordances) > 0 and
                    len(graph.spatial_reasoning_chain) > 0):
                    valid_graphs += 1

            validation_results['checks'].append({
                'check': 'scene_graph_structure',
                'valid_graphs': valid_graphs,
                'total_graphs': len(self.scene_graphs),
                'passed': valid_graphs == len(self.scene_graphs)
            })

            # Save validation results
            validation_file = self.output_dir / "validation" / "validation_results.json"
            with open(validation_file, 'w') as f:
                json.dump(validation_results, f, indent=2)

            # Check overall validation status
            all_passed = all(check['passed'] for check in validation_results['checks'])

            if all_passed:
                self.log("All validation checks passed")
            else:
                self.log("Some validation checks failed - see validation_results.json", "WARNING")

            return all_passed

        except Exception as e:
            self.log(f"Error during validation: {e}", "ERROR")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Scene Graph Generation Pipeline")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation checks on outputs")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--output-dir", default="outputs",
                        help="Output directory (default: outputs)")

    args = parser.parse_args()

    # Check if we're in the right directory
    if not Path("data_examples").exists():
        print("ERROR: data_examples directory not found.")
        print("Make sure you're running this script from the alignment directory.")
        print("Expected structure: /home/jiachen/scratch/SceneFun3D/alignment/")
        sys.exit(1)

    # Create and run pipeline
    pipeline = SceneGraphPipeline(
        output_dir=args.output_dir,
        verbose=args.verbose
    )

    success = pipeline.run_pipeline(validate=args.validate)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()