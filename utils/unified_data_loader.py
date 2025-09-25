"""
Unified Data Loader - V2.0 Pipeline

A configurable data loading interface that wraps the official SceneFun3D DataParser
and provides unified access to all scene data with robust error handling.

Key Features:
- Works with any data root path
- Wraps official SceneFun3D DataParser
- Graceful error handling for missing data
- Caching for large datasets
- Consistent data format across pipeline
"""

import sys
import os
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
import warnings

# Suppress Open3D warnings
warnings.filterwarnings('ignore')


@dataclass
class SceneData:
    """Container for all scene data."""
    visit_id: str
    video_id: str
    data_root: Path

    # Core data
    laser_scan: Optional[Any] = None
    arkit_mesh: Optional[Any] = None
    arkit_point_cloud: Optional[Any] = None
    transform_matrix: Optional[np.ndarray] = None

    # Annotation data
    annotations: Optional[List[Dict]] = None
    descriptions: Optional[List[Dict]] = None
    motions: Optional[List[Dict]] = None

    # Derived data
    annotation_lookup: Optional[Dict[str, Dict]] = None
    motion_lookup: Optional[Dict[str, Dict]] = None

    # Metadata
    data_stats: Dict[str, Any] = None
    load_errors: List[str] = None

    def __post_init__(self):
        if self.load_errors is None:
            self.load_errors = []
        if self.data_stats is None:
            self.data_stats = {}


class UnifiedDataLoader:
    """
    Unified data loading interface for SceneFun3D pipeline.

    Provides consistent access to SceneFun3D data regardless of data root location.
    """

    def __init__(self, data_root: Union[str, Path], visit_id: str, video_id: str,
                 auto_find_scenefun3d: bool = True, cache_enabled: bool = True):
        """
        Initialize the unified data loader.

        Args:
            data_root: Root directory containing SceneFun3D data
            visit_id: Visit/scene identifier
            video_id: Video sequence identifier
            auto_find_scenefun3d: Automatically locate SceneFun3D toolkit
            cache_enabled: Enable caching for large data files
        """
        self.data_root = Path(data_root).resolve()
        self.visit_id = str(visit_id)
        self.video_id = str(video_id)
        self.cache_enabled = cache_enabled
        self.logger = self._setup_logger()

        # Data cache
        self._cache = {}

        # Verify data root exists
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root does not exist: {self.data_root}")

        # Initialize SceneFun3D DataParser
        self.parser = self._initialize_dataparser(auto_find_scenefun3d)

        self.logger.info(f"Initialized UnifiedDataLoader")
        self.logger.info(f"  Data root: {self.data_root}")
        self.logger.info(f"  Visit ID: {self.visit_id}")
        self.logger.info(f"  Video ID: {self.video_id}")

    def _setup_logger(self) -> logging.Logger:
        """Setup logger for data loading operations."""
        logger = logging.getLogger(f"UnifiedDataLoader_{self.visit_id}_{self.video_id}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _initialize_dataparser(self, auto_find: bool) -> Any:
        """Initialize the official SceneFun3D DataParser."""
        if auto_find:
            # Try to find scenefun3d directory
            scenefun3d_paths = [
                Path("/home/jiachen/scratch/SceneFun3D/scenefun3d"),  # Explicit path
                Path(__file__).parent.parent.parent / "scenefun3d",  # ../../scenefun3d
                Path(__file__).parent.parent / "scenefun3d",         # ../scenefun3d
                Path.home() / "scenefun3d",                          # ~/scenefun3d
            ]

            scenefun3d_root = None
            for path in scenefun3d_paths:
                if path.exists() and (path / "utils" / "data_parser.py").exists():
                    scenefun3d_root = path
                    break

            if not scenefun3d_root:
                raise ImportError(
                    "SceneFun3D toolkit not found. Please ensure it's installed or set auto_find_scenefun3d=False"
                )

            # Setup imports with preserved working directory
            original_cwd = os.getcwd()
            original_path = sys.path.copy()

            try:
                os.chdir(str(scenefun3d_root))
                sys.path.insert(0, str(scenefun3d_root))
                from utils.data_parser import DataParser
                self.logger.info(f"Imported SceneFun3D DataParser from {scenefun3d_root}")

                # Store the DataParser class for later use
                self._DataParser = DataParser

            finally:
                os.chdir(original_cwd)
                # Don't restore sys.path to keep the import working
        else:
            # Assume DataParser is already available
            from utils.data_parser import DataParser
            self._DataParser = DataParser

        return self._DataParser(str(self.data_root))

    def load_all_data(self, include_mesh: bool = True, include_point_cloud: bool = False) -> SceneData:
        """
        Load all available scene data.

        Args:
            include_mesh: Load ARKit mesh (can be large)
            include_point_cloud: Load ARKit point cloud (can be large)

        Returns:
            SceneData object containing all loaded data
        """
        self.logger.info("Loading all scene data...")

        scene_data = SceneData(
            visit_id=self.visit_id,
            video_id=self.video_id,
            data_root=self.data_root
        )

        # Load core data components
        loaders = [
            ("laser_scan", self._load_laser_scan),
            ("transform_matrix", self._load_transform_matrix),
            ("annotations", self._load_annotations),
            ("descriptions", self._load_descriptions),
            ("motions", self._load_motions),
        ]

        if include_mesh:
            loaders.append(("arkit_mesh", self._load_arkit_mesh))

        if include_point_cloud:
            loaders.append(("arkit_point_cloud", self._load_arkit_point_cloud))

        # Execute loaders
        for attr_name, loader_func in loaders:
            try:
                data = loader_func()
                setattr(scene_data, attr_name, data)
                self.logger.info(f"✅ Loaded {attr_name}")
            except Exception as e:
                error_msg = f"Failed to load {attr_name}: {str(e)}"
                scene_data.load_errors.append(error_msg)
                self.logger.warning(f"❌ {error_msg}")

        # Create lookup dictionaries
        if scene_data.annotations:
            scene_data.annotation_lookup = {
                ann['annot_id']: ann for ann in scene_data.annotations
            }

        if scene_data.motions:
            scene_data.motion_lookup = {
                motion['annot_id']: motion for motion in scene_data.motions
            }

        # Generate statistics
        scene_data.data_stats = self._generate_data_stats(scene_data)

        self.logger.info(f"Data loading completed. Errors: {len(scene_data.load_errors)}")
        return scene_data

    def _load_laser_scan(self):
        """Load laser scan point cloud."""
        cache_key = "laser_scan"
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        laser_scan = self.parser.get_laser_scan(self.visit_id)

        if self.cache_enabled:
            self._cache[cache_key] = laser_scan

        return laser_scan

    def _load_arkit_mesh(self):
        """Load ARKit mesh reconstruction."""
        cache_key = "arkit_mesh"
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        arkit_mesh = self.parser.get_arkit_reconstruction(
            self.visit_id, self.video_id, format="mesh"
        )

        if self.cache_enabled:
            self._cache[cache_key] = arkit_mesh

        return arkit_mesh

    def _load_arkit_point_cloud(self):
        """Load ARKit point cloud reconstruction."""
        cache_key = "arkit_point_cloud"
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        arkit_pc = self.parser.get_arkit_reconstruction(
            self.visit_id, self.video_id, format="point_cloud"
        )

        if self.cache_enabled:
            self._cache[cache_key] = arkit_pc

        return arkit_pc

    def _load_transform_matrix(self) -> np.ndarray:
        """Load coordinate transformation matrix."""
        cache_key = "transform_matrix"
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        transform = self.parser.get_transform(self.visit_id, self.video_id)

        # Validate transformation matrix
        if transform.shape != (4, 4):
            raise ValueError(f"Expected 4x4 transformation matrix, got {transform.shape}")

        # Check if matrix is invertible
        try:
            np.linalg.inv(transform)
        except np.linalg.LinAlgError:
            raise ValueError("Transformation matrix is not invertible")

        if self.cache_enabled:
            self._cache[cache_key] = transform

        return transform

    def _load_annotations(self) -> List[Dict]:
        """Load affordance annotations."""
        annotations = self.parser.get_annotations(self.visit_id)

        # Validate annotation structure
        for ann in annotations:
            if 'annot_id' not in ann:
                raise ValueError("Annotation missing 'annot_id' field")
            if 'indices' not in ann:
                raise ValueError(f"Annotation {ann['annot_id']} missing 'indices' field")

        return annotations

    def _load_descriptions(self) -> List[Dict]:
        """Load task descriptions."""
        descriptions = self.parser.get_descriptions(self.visit_id)

        # Validate description structure
        for desc in descriptions:
            if 'description' not in desc:
                raise ValueError("Task description missing 'description' field")
            if 'annot_ids' not in desc:
                raise ValueError("Task description missing 'annot_ids' field")

        return descriptions

    def _load_motions(self) -> List[Dict]:
        """Load motion annotations."""
        motions = self.parser.get_motions(self.visit_id)

        # Validate motion structure
        for motion in motions:
            if 'annot_id' not in motion:
                raise ValueError("Motion annotation missing 'annot_id' field")
            if 'motion_type' not in motion:
                raise ValueError(f"Motion {motion['annot_id']} missing 'motion_type' field")

        return motions

    def _generate_data_stats(self, scene_data: SceneData) -> Dict[str, Any]:
        """Generate statistics about loaded data."""
        stats = {
            "load_success": {
                "laser_scan": scene_data.laser_scan is not None,
                "arkit_mesh": scene_data.arkit_mesh is not None,
                "arkit_point_cloud": scene_data.arkit_point_cloud is not None,
                "transform_matrix": scene_data.transform_matrix is not None,
                "annotations": scene_data.annotations is not None,
                "descriptions": scene_data.descriptions is not None,
                "motions": scene_data.motions is not None,
            }
        }

        # Data counts
        if scene_data.laser_scan:
            stats["laser_scan_points"] = len(scene_data.laser_scan.points)

        if scene_data.arkit_mesh:
            stats["arkit_mesh_vertices"] = len(scene_data.arkit_mesh.vertices)
            stats["arkit_mesh_triangles"] = len(scene_data.arkit_mesh.triangles)

        if scene_data.arkit_point_cloud:
            stats["arkit_pc_points"] = len(scene_data.arkit_point_cloud.points)

        if scene_data.annotations:
            stats["annotation_count"] = len(scene_data.annotations)
            stats["total_annotation_points"] = sum(
                len(ann.get('indices', [])) for ann in scene_data.annotations
            )

        if scene_data.descriptions:
            stats["task_count"] = len(scene_data.descriptions)

        if scene_data.motions:
            stats["motion_count"] = len(scene_data.motions)
            stats["motion_types"] = list(set(
                motion.get('motion_type', 'unknown') for motion in scene_data.motions
            ))

        stats["error_count"] = len(scene_data.load_errors)

        return stats

    def get_affordance_points(self, annotation_id: str, scene_data: SceneData) -> Optional[np.ndarray]:
        """
        Get 3D coordinates of affordance points in SceneFun3D coordinate system.

        Args:
            annotation_id: Annotation identifier
            scene_data: Loaded scene data

        Returns:
            Nx3 array of 3D points in SceneFun3D coordinates, or None if not found
        """
        if not scene_data.annotation_lookup or annotation_id not in scene_data.annotation_lookup:
            self.logger.warning(f"Annotation {annotation_id} not found")
            return None

        if scene_data.laser_scan is None:
            self.logger.warning("Laser scan not loaded, cannot get affordance points")
            return None

        annotation = scene_data.annotation_lookup[annotation_id]
        indices = annotation.get('indices', [])

        if not indices:
            self.logger.warning(f"No point indices for annotation {annotation_id}")
            return None

        # Get laser scan points (already in SceneFun3D coordinate system)
        laser_points = np.asarray(scene_data.laser_scan.points)

        # Validate indices
        max_index = len(laser_points) - 1
        valid_indices = [i for i in indices if 0 <= i <= max_index]

        if len(valid_indices) != len(indices):
            self.logger.warning(f"Some invalid indices in annotation {annotation_id}")

        if not valid_indices:
            return None

        affordance_points = laser_points[valid_indices]
        return affordance_points

    def validate_data_consistency(self, scene_data: SceneData) -> Dict[str, Any]:
        """
        Validate consistency between different data components.

        Args:
            scene_data: Loaded scene data

        Returns:
            Dictionary with consistency validation results
        """
        validation_results = {}

        if not scene_data.annotations or not scene_data.descriptions or not scene_data.motions:
            validation_results["error"] = "Missing required data components"
            return validation_results

        # Check annotation ID consistency
        annotation_ids = set(ann['annot_id'] for ann in scene_data.annotations)

        description_annotation_ids = set()
        for desc in scene_data.descriptions:
            description_annotation_ids.update(desc.get('annot_ids', []))

        motion_annotation_ids = set(motion['annot_id'] for motion in scene_data.motions)

        # Coverage analysis
        description_coverage = len(description_annotation_ids & annotation_ids) / len(annotation_ids)
        motion_coverage = len(motion_annotation_ids & annotation_ids) / len(annotation_ids)

        validation_results.update({
            "annotation_count": len(annotation_ids),
            "description_coverage": description_coverage,
            "motion_coverage": motion_coverage,
            "missing_in_descriptions": list(annotation_ids - description_annotation_ids),
            "missing_in_motions": list(annotation_ids - motion_annotation_ids),
        })

        # Point index validation (sample check)
        if scene_data.laser_scan:
            laser_scan_size = len(scene_data.laser_scan.points)
            invalid_annotations = []

            for ann in scene_data.annotations[:10]:  # Check first 10
                indices = ann.get('indices', [])
                if indices and max(indices) >= laser_scan_size:
                    invalid_annotations.append(ann['annot_id'])

            validation_results["invalid_point_indices"] = invalid_annotations

        return validation_results

    def clear_cache(self):
        """Clear data cache to free memory."""
        self._cache.clear()
        self.logger.info("Data cache cleared")

    def get_data_summary(self, scene_data: SceneData) -> str:
        """Get human-readable summary of loaded data."""
        summary_lines = [
            f"Scene Data Summary - Visit {self.visit_id}, Video {self.video_id}",
            "=" * 60
        ]

        if scene_data.data_stats:
            stats = scene_data.data_stats

            # Success summary
            success_items = [k for k, v in stats.get("load_success", {}).items() if v]
            summary_lines.append(f"✅ Loaded: {', '.join(success_items)}")

            # Data counts
            if "laser_scan_points" in stats:
                summary_lines.append(f"📊 Laser scan: {stats['laser_scan_points']:,} points")
            if "arkit_mesh_vertices" in stats:
                summary_lines.append(f"📊 ARKit mesh: {stats['arkit_mesh_vertices']:,} vertices, {stats['arkit_mesh_triangles']:,} triangles")
            if "annotation_count" in stats:
                summary_lines.append(f"📊 Annotations: {stats['annotation_count']} ({stats.get('total_annotation_points', 0):,} points)")
            if "task_count" in stats:
                summary_lines.append(f"📊 Tasks: {stats['task_count']}")
            if "motion_types" in stats:
                summary_lines.append(f"📊 Motion types: {', '.join(stats['motion_types'])}")

        if scene_data.load_errors:
            summary_lines.append(f"❌ Errors: {len(scene_data.load_errors)}")
            for error in scene_data.load_errors[:3]:  # Show first 3
                summary_lines.append(f"   - {error}")

        return "\n".join(summary_lines)