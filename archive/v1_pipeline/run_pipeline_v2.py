#!/usr/bin/env python3
"""
V2.0 Scene Graph Generation Pipeline

Main pipeline that orchestrates all v2.0 components:
- UnifiedDataLoader for configurable data access
- Enhanced coordinate transformation (ARKit -> SceneFun3D)
- Spatial relationship scoring with no fallback logic
- Scene graph generation with floating affordances

Usage:
    python run_pipeline_v2.py --data-root /path/to/data --arkitscenes-file /path/to/arkit.json
    python run_pipeline_v2.py --config config_v2.yaml
    python run_pipeline_v2.py --help
"""

import sys
import os
import argparse
import yaml
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add utils and scripts to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "utils"))
sys.path.append(str(Path(__file__).parent / "scripts"))

try:
    from utils.unified_data_loader import UnifiedDataLoader
    from utils.enhanced_coordinate_transform import EnhancedCoordinateTransformer
    from utils.spatial_scorer import SpatialScorer
    from scripts.scene_graph_builder_v2 import SceneGraphBuilderV2, save_scene_graph
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("   Make sure all v2.0 components are available")
    sys.exit(1)


class PipelineV2:
    """
    V2.0 Scene Graph Generation Pipeline.

    Coordinates all components to generate accurate scene graphs with:
    - Official SceneFun3D DataParser integration
    - ARKit to SceneFun3D coordinate transformation
    - Confidence-based spatial relationships
    - Floating affordances support
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize pipeline with configuration."""
        self.config = config

        # Extract configuration first (needed for logger setup)
        self.data_root = Path(config['data']['root_path'])
        self.visit_id = config['data']['visit_id']
        self.video_id = config['data']['video_id']
        self.arkitscenes_file = Path(config['data']['arkitscenes_file'])
        self.output_dir = Path(config['output']['directory'])

        # Now setup logger (after output_dir is defined)
        self.logger = self._setup_logger()

        # Pipeline parameters
        self.confidence_threshold = config.get('spatial_analysis', {}).get('confidence_threshold', 0.3)
        self.allow_floating = config.get('spatial_analysis', {}).get('allow_floating', True)

        # Validation
        self._validate_configuration()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.data_loader = None
        self.coordinate_transformer = None
        self.spatial_scorer = None
        self.scene_graph_builder = None

        # Statistics
        self.stats = {
            'start_time': None,
            'end_time': None,
            'duration_seconds': 0,
            'data_loading_time': 0,
            'transformation_time': 0,
            'spatial_analysis_time': 0,
            'scene_graph_time': 0,
            'objects_loaded': 0,
            'affordances_extracted': 0,
            'relationships_scored': 0,
            'scene_graphs_generated': 0,
            'confident_relationships': 0,
            'floating_affordances': 0,
            'errors': []
        }

        self.logger.info("🚀 V2.0 Pipeline initialized")
        self.logger.info(f"   📁 Data root: {self.data_root}")
        self.logger.info(f"   🏠 Visit ID: {self.visit_id}")
        self.logger.info(f"   🎥 Video ID: {self.video_id}")
        self.logger.info(f"   🎯 Confidence threshold: {self.confidence_threshold}")

    def _setup_logger(self) -> logging.Logger:
        """Setup pipeline logger."""
        logger = logging.getLogger("PipelineV2")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # File handler
            log_file = self.output_dir / 'pipeline_v2.log'
            self.output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            file_handler = logging.FileHandler(str(log_file))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def _validate_configuration(self):
        """Validate pipeline configuration."""
        errors = []

        if not self.data_root.exists():
            errors.append(f"Data root does not exist: {self.data_root}")

        if not self.arkitscenes_file.exists():
            errors.append(f"ARKitScenes file does not exist: {self.arkitscenes_file}")

        if not (0.0 <= self.confidence_threshold <= 1.0):
            errors.append(f"Confidence threshold must be in [0, 1], got {self.confidence_threshold}")

        if errors:
            for error in errors:
                self.logger.error(error)
            raise ValueError(f"Configuration validation failed: {errors}")

    def initialize_components(self):
        """Initialize pipeline components."""
        self.logger.info("🔧 Initializing pipeline components...")

        try:
            # Data loader
            self.data_loader = UnifiedDataLoader(
                str(self.data_root),
                self.visit_id,
                self.video_id,
                cache_enabled=True
            )

            # Scene graph builder (creates other components internally)
            self.scene_graph_builder = SceneGraphBuilderV2(
                confidence_threshold=self.confidence_threshold,
                logger=self.logger
            )

            self.logger.info("✅ All components initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize components: {e}"
            self.logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            raise

    def validate_data_availability(self) -> bool:
        """Validate that all required data is available."""
        self.logger.info("🔍 Validating data availability...")

        try:
            # Test data loading
            scene_data = self.data_loader.load_all_data(include_mesh=False, include_point_cloud=False)

            # Check for critical errors
            if not scene_data.transform_matrix is not None:
                self.logger.error("❌ Transform matrix not available")
                return False

            if not scene_data.annotations:
                self.logger.error("❌ No annotations available")
                return False

            if not scene_data.descriptions:
                self.logger.error("❌ No task descriptions available")
                return False

            # Check ARKitScenes file
            if not self.arkitscenes_file.exists():
                self.logger.error(f"❌ ARKitScenes file not found: {self.arkitscenes_file}")
                return False

            # Log data statistics
            self.logger.info(f"✅ Data validation passed:")
            self.logger.info(f"   📊 Transform matrix: {scene_data.transform_matrix.shape}")
            self.logger.info(f"   📊 Annotations: {len(scene_data.annotations)}")
            self.logger.info(f"   📊 Task descriptions: {len(scene_data.descriptions)}")
            self.logger.info(f"   📊 Load errors: {len(scene_data.load_errors)}")

            if scene_data.load_errors:
                self.logger.warning("⚠️ Some data loading errors occurred:")
                for error in scene_data.load_errors[:3]:  # Show first 3
                    self.logger.warning(f"   - {error}")

            return True

        except Exception as e:
            error_msg = f"Data validation failed: {e}"
            self.logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            return False

    def run_pipeline(self) -> bool:
        """Run the complete v2.0 pipeline."""
        self.logger.info("🏁 Starting V2.0 Scene Graph Generation Pipeline")
        self.stats['start_time'] = time.time()

        try:
            # Step 1: Initialize components
            self.initialize_components()

            # Step 2: Validate data
            if not self.validate_data_availability():
                return False

            # Step 3: Build scene graphs
            self.logger.info("🏗️ Building scene graphs...")
            scene_graph_start = time.time()

            scene_graphs = self.scene_graph_builder.build_all_task_graphs(
                str(self.data_root),
                str(self.arkitscenes_file),
                self.visit_id,
                self.video_id
            )

            self.stats['scene_graph_time'] = time.time() - scene_graph_start
            self.stats['scene_graphs_generated'] = len(scene_graphs)

            if not scene_graphs:
                self.logger.error("❌ No scene graphs generated")
                return False

            # Step 4: Save scene graphs
            self.logger.info("💾 Saving scene graphs...")
            self.save_scene_graphs(scene_graphs)

            # Step 5: Generate statistics
            self.calculate_statistics(scene_graphs)

            # Step 6: Save pipeline results
            self.save_pipeline_results()

            self.stats['end_time'] = time.time()
            self.stats['duration_seconds'] = self.stats['end_time'] - self.stats['start_time']

            self.logger.info("🎉 Pipeline completed successfully!")
            self.print_summary()

            return True

        except Exception as e:
            error_msg = f"Pipeline execution failed: {e}"
            self.logger.error(error_msg)
            self.stats['errors'].append(error_msg)

            self.stats['end_time'] = time.time()
            if self.stats['start_time']:
                self.stats['duration_seconds'] = self.stats['end_time'] - self.stats['start_time']

            return False

    def save_scene_graphs(self, scene_graphs: List[Any]):
        """Save scene graphs to JSON files."""
        graphs_dir = self.output_dir / "scene_graphs"
        graphs_dir.mkdir(exist_ok=True)

        for i, graph in enumerate(scene_graphs):
            try:
                # Generate filename
                task_name = graph.task_description.lower().replace(' ', '_').replace(',', '').replace('.', '')
                filename = f"{self.visit_id}_task_{i+1:02d}_{task_name}.json"
                output_file = graphs_dir / filename

                # Save graph
                save_scene_graph(graph, str(output_file))

                self.logger.info(f"   📝 Saved: {filename}")

            except Exception as e:
                error_msg = f"Failed to save scene graph {i+1}: {e}"
                self.logger.error(error_msg)
                self.stats['errors'].append(error_msg)

    def calculate_statistics(self, scene_graphs: List[Any]):
        """Calculate pipeline statistics."""
        total_objects = 0
        total_affordances = 0
        total_confident = 0
        total_floating = 0

        for graph in scene_graphs:
            total_objects += len(graph.object_nodes)
            total_affordances += len(graph.affordance_nodes)
            total_confident += len([n for n in graph.affordance_nodes.values() if not n.is_floating])
            total_floating += len(graph.floating_affordances)

        self.stats.update({
            'objects_loaded': total_objects,
            'affordances_extracted': total_affordances,
            'confident_relationships': total_confident,
            'floating_affordances': total_floating
        })

    def save_pipeline_results(self):
        """Save complete pipeline results and statistics."""
        results = {
            'pipeline_version': '2.0',
            'configuration': self.config,
            'statistics': self.stats,
            'summary': {
                'success': len(self.stats['errors']) == 0,
                'total_duration': self.stats['duration_seconds'],
                'scene_graphs_generated': self.stats['scene_graphs_generated'],
                'confident_relationships': self.stats['confident_relationships'],
                'floating_affordances': self.stats['floating_affordances'],
                'error_count': len(self.stats['errors'])
            }
        }

        # Save results
        results_file = self.output_dir / "pipeline_results_v2.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"📊 Results saved to: {results_file}")

    def print_summary(self):
        """Print pipeline execution summary."""
        print("\n" + "="*60)
        print("🎯 V2.0 PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"✅ Duration: {self.stats['duration_seconds']:.1f} seconds")
        print(f"✅ Scene graphs: {self.stats['scene_graphs_generated']}")
        print(f"✅ Objects: {self.stats['objects_loaded']}")
        print(f"✅ Affordances: {self.stats['affordances_extracted']}")
        print(f"✅ Confident relationships: {self.stats['confident_relationships']}")
        print(f"✅ Floating affordances: {self.stats['floating_affordances']}")

        if self.stats['errors']:
            print(f"⚠️ Errors: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:3]:
                print(f"   - {error}")

        print(f"\n📁 Output directory: {self.output_dir}")
        print("="*60)


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"❌ Failed to load config file: {e}")
        sys.exit(1)


def create_default_config(data_root: str, arkitscenes_file: str,
                         visit_id: str = "422203", video_id: str = "42445781") -> Dict[str, Any]:
    """Create default configuration."""
    return {
        'data': {
            'root_path': data_root,
            'visit_id': visit_id,
            'video_id': video_id,
            'arkitscenes_file': arkitscenes_file
        },
        'spatial_analysis': {
            'confidence_threshold': 0.3,
            'allow_floating': True,
            'distance_weight': 0.4,
            'overlap_weight': 0.6,
            'max_distance': 500.0
        },
        'output': {
            'directory': 'outputs_v2',
            'save_intermediate': False
        }
    }


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description="V2.0 Scene Graph Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline_v2.py --data-root /path/to/data --arkitscenes-file /path/to/arkit.json
  python run_pipeline_v2.py --config config_v2.yaml
  python run_pipeline_v2.py --data-root /path/to/data --arkitscenes-file /path/to/arkit.json --confidence-threshold 0.4
        """
    )

    parser.add_argument("--config", help="YAML configuration file")
    parser.add_argument("--data-root", help="SceneFun3D data root directory")
    parser.add_argument("--arkitscenes-file", help="ARKitScenes annotation JSON file")
    parser.add_argument("--visit-id", default="422203", help="Visit ID (default: 422203)")
    parser.add_argument("--video-id", default="42445781", help="Video ID (default: 42445781)")
    parser.add_argument("--output-dir", default="outputs_v2", help="Output directory (default: outputs_v2)")
    parser.add_argument("--confidence-threshold", type=float, default=0.3,
                       help="Confidence threshold for relationships (default: 0.3)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = load_config(args.config)
    elif args.data_root and args.arkitscenes_file:
        config = create_default_config(args.data_root, args.arkitscenes_file,
                                     args.visit_id, args.video_id)
        # Override with command line arguments
        if args.output_dir:
            config['output']['directory'] = args.output_dir
        if args.confidence_threshold:
            config['spatial_analysis']['confidence_threshold'] = args.confidence_threshold
    else:
        parser.error("Either --config or both --data-root and --arkitscenes-file are required")

    try:
        # Create and run pipeline
        pipeline = PipelineV2(config)
        success = pipeline.run_pipeline()

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n⏹️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Pipeline failed with fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()