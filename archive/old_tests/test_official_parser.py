#!/usr/bin/env python3
"""
Test Official SceneFun3D DataParser - V2.0 Pipeline

This script validates the official SceneFun3D DataParser functionality with configurable data paths.
It ensures compatibility and data consistency before integrating into the v2 pipeline.

Usage:
    python test_official_parser.py --data-root /path/to/scenefun3d/data --visit-id 422203 --video-id 42445781
    python test_official_parser.py --help
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path
import json

def setup_scenefun3d_imports():
    """Setup SceneFun3D imports with dynamic path handling."""
    # Try to find scenefun3d directory
    current_dir = Path(__file__).parent
    scenefun3d_paths = [
        current_dir.parent / "scenefun3d",  # ../scenefun3d
        current_dir / "scenefun3d",         # ./scenefun3d
        Path.home() / "scenefun3d",         # ~/scenefun3d
    ]

    scenefun3d_root = None
    for path in scenefun3d_paths:
        if path.exists() and (path / "utils" / "data_parser.py").exists():
            scenefun3d_root = path
            break

    if not scenefun3d_root:
        print("❌ SceneFun3D toolkit not found. Please ensure it's installed.")
        print("   Searched paths:")
        for path in scenefun3d_paths:
            print(f"   - {path}")
        sys.exit(1)

    # Change working directory temporarily for imports
    original_cwd = os.getcwd()
    os.chdir(str(scenefun3d_root))
    sys.path.insert(0, str(scenefun3d_root))

    try:
        from utils.data_parser import DataParser
        print(f"✅ Successfully imported SceneFun3D DataParser from {scenefun3d_root}")
        return DataParser, original_cwd
    except ImportError as e:
        print(f"❌ Failed to import SceneFun3D DataParser: {e}")
        sys.exit(1)
    finally:
        os.chdir(original_cwd)


class OfficialParserValidator:
    """Comprehensive validator for SceneFun3D DataParser."""

    def __init__(self, data_root: str, visit_id: str, video_id: str):
        """Initialize validator with configurable paths."""
        self.data_root = Path(data_root).resolve()
        self.visit_id = str(visit_id)
        self.video_id = str(video_id)

        print(f"🔧 Initializing validator with:")
        print(f"   📁 Data root: {self.data_root}")
        print(f"   🏠 Visit ID: {self.visit_id}")
        print(f"   🎥 Video ID: {self.video_id}")

        # Verify data root exists
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root does not exist: {self.data_root}")

        # Initialize DataParser
        DataParser, _ = setup_scenefun3d_imports()
        self.parser = DataParser(str(self.data_root))

        self.test_results = {}

    def validate_data_structure(self):
        """Validate expected data structure exists."""
        print("\n📋 Validating data structure...")

        visit_dir = self.data_root / self.visit_id
        video_dir = visit_dir / self.video_id

        expected_files = {
            "laser_scan": visit_dir / f"{self.visit_id}_laser_scan.ply",
            "annotations": visit_dir / f"{self.visit_id}_annotations.json",
            "descriptions": visit_dir / f"{self.visit_id}_descriptions.json",
            "motions": visit_dir / f"{self.visit_id}_motions.json",
            "arkit_mesh": video_dir / f"{self.video_id}_arkit_mesh.ply",
            "transform": video_dir / f"{self.video_id}_transform.npy",
        }

        structure_valid = True
        for name, path in expected_files.items():
            if path.exists():
                print(f"   ✅ {name}: {path}")
            else:
                print(f"   ❌ {name}: MISSING - {path}")
                structure_valid = False

        self.test_results["data_structure"] = structure_valid
        return structure_valid

    def test_laser_scan_loading(self):
        """Test laser scan loading."""
        print("\n🎯 Testing laser scan loading...")

        try:
            laser_scan = self.parser.get_laser_scan(self.visit_id)
            point_count = len(laser_scan.points)
            has_colors = len(laser_scan.colors) > 0

            print(f"   ✅ Loaded laser scan: {point_count:,} points")
            print(f"   🎨 Has colors: {has_colors}")

            # Get point bounds
            points = np.asarray(laser_scan.points)
            bounds = {
                "min": points.min(axis=0).tolist(),
                "max": points.max(axis=0).tolist(),
                "range": (points.max(axis=0) - points.min(axis=0)).tolist()
            }
            print(f"   📏 Point bounds: {bounds['min']} to {bounds['max']}")

            self.test_results["laser_scan"] = {
                "success": True,
                "point_count": point_count,
                "has_colors": has_colors,
                "bounds": bounds
            }
            return True

        except Exception as e:
            print(f"   ❌ Failed to load laser scan: {e}")
            self.test_results["laser_scan"] = {"success": False, "error": str(e)}
            return False

    def test_arkit_reconstruction_loading(self):
        """Test ARKit reconstruction loading."""
        print("\n🏗️ Testing ARKit reconstruction loading...")

        results = {"point_cloud": False, "mesh": False}

        # Test point cloud format
        try:
            arkit_pc = self.parser.get_arkit_reconstruction(self.visit_id, self.video_id, format="point_cloud")
            pc_point_count = len(arkit_pc.points)
            print(f"   ✅ Point cloud: {pc_point_count:,} points")
            results["point_cloud"] = True
        except Exception as e:
            print(f"   ❌ Point cloud failed: {e}")

        # Test mesh format
        try:
            arkit_mesh = self.parser.get_arkit_reconstruction(self.visit_id, self.video_id, format="mesh")
            vertex_count = len(arkit_mesh.vertices)
            triangle_count = len(arkit_mesh.triangles)
            print(f"   ✅ Mesh: {vertex_count:,} vertices, {triangle_count:,} triangles")
            results["mesh"] = True
        except Exception as e:
            print(f"   ❌ Mesh failed: {e}")

        self.test_results["arkit_reconstruction"] = results
        return any(results.values())

    def test_transformation_matrix(self):
        """Test transformation matrix loading and properties."""
        print("\n🔄 Testing transformation matrix...")

        try:
            transform = self.parser.get_transform(self.visit_id, self.video_id)
            print(f"   ✅ Loaded transform: {transform.shape}")

            # Validate transformation properties
            is_homogeneous = transform.shape == (4, 4)
            determinant = np.linalg.det(transform[:3, :3]) if is_homogeneous else np.nan

            print(f"   📊 Is 4x4 homogeneous: {is_homogeneous}")
            if is_homogeneous:
                print(f"   📊 Rotation determinant: {determinant:.6f}")
                print(f"   📊 Translation: {transform[:3, 3]}")

                # Test invertibility
                try:
                    inverse_transform = np.linalg.inv(transform)
                    print(f"   ✅ Matrix is invertible")
                except:
                    print(f"   ❌ Matrix is not invertible")

            self.test_results["transform"] = {
                "success": True,
                "shape": transform.shape,
                "determinant": float(determinant) if not np.isnan(determinant) else None,
                "is_homogeneous": is_homogeneous
            }
            return True

        except Exception as e:
            print(f"   ❌ Failed to load transform: {e}")
            self.test_results["transform"] = {"success": False, "error": str(e)}
            return False

    def test_annotations_loading(self):
        """Test annotations loading."""
        print("\n📝 Testing annotations loading...")

        try:
            annotations = self.parser.get_annotations(self.visit_id)
            print(f"   ✅ Loaded annotations: {len(annotations)} items")

            # Analyze annotation structure
            annotation_ids = [ann.get('annot_id', 'unknown') for ann in annotations]
            point_counts = [len(ann.get('indices', [])) for ann in annotations]

            print(f"   📊 Annotation IDs: {annotation_ids[:5]}...")  # Show first 5
            print(f"   📊 Point counts: min={min(point_counts)}, max={max(point_counts)}, total={sum(point_counts)}")

            self.test_results["annotations"] = {
                "success": True,
                "count": len(annotations),
                "total_points": sum(point_counts),
                "annotation_ids": annotation_ids
            }
            return True

        except Exception as e:
            print(f"   ❌ Failed to load annotations: {e}")
            self.test_results["annotations"] = {"success": False, "error": str(e)}
            return False

    def test_descriptions_loading(self):
        """Test task descriptions loading."""
        print("\n📖 Testing task descriptions loading...")

        try:
            descriptions = self.parser.get_descriptions(self.visit_id)
            print(f"   ✅ Loaded descriptions: {len(descriptions)} tasks")

            # Show task descriptions
            for i, desc in enumerate(descriptions[:3]):  # Show first 3
                task_text = desc.get('description', 'No description')
                annot_ids = desc.get('annot_ids', [])
                print(f"   📋 Task {i+1}: '{task_text}' (annotations: {len(annot_ids)})")

            self.test_results["descriptions"] = {
                "success": True,
                "count": len(descriptions),
                "tasks": [desc.get('description', '') for desc in descriptions]
            }
            return True

        except Exception as e:
            print(f"   ❌ Failed to load descriptions: {e}")
            self.test_results["descriptions"] = {"success": False, "error": str(e)}
            return False

    def test_motions_loading(self):
        """Test motion annotations loading."""
        print("\n🎯 Testing motion annotations loading...")

        try:
            motions = self.parser.get_motions(self.visit_id)
            print(f"   ✅ Loaded motions: {len(motions)} items")

            # Analyze motion types
            motion_types = set()
            for motion in motions:
                motion_type = motion.get('motion_type', 'unknown')
                motion_types.add(motion_type)

            print(f"   📊 Motion types: {sorted(motion_types)}")

            self.test_results["motions"] = {
                "success": True,
                "count": len(motions),
                "motion_types": list(motion_types)
            }
            return True

        except Exception as e:
            print(f"   ❌ Failed to load motions: {e}")
            self.test_results["motions"] = {"success": False, "error": str(e)}
            return False

    def test_data_consistency(self):
        """Test consistency between different data components."""
        print("\n🔍 Testing data consistency...")

        try:
            # Load all components
            annotations = self.parser.get_annotations(self.visit_id)
            descriptions = self.parser.get_descriptions(self.visit_id)
            motions = self.parser.get_motions(self.visit_id)
            laser_scan = self.parser.get_laser_scan(self.visit_id)

            # Check annotation ID consistency
            annotation_ids = set(ann['annot_id'] for ann in annotations)
            description_annotation_ids = set()
            for desc in descriptions:
                description_annotation_ids.update(desc.get('annot_ids', []))
            motion_annotation_ids = set(motion['annot_id'] for motion in motions)

            # Check consistency
            missing_in_descriptions = annotation_ids - description_annotation_ids
            missing_in_motions = annotation_ids - motion_annotation_ids

            print(f"   📊 Annotation IDs: {len(annotation_ids)} total")
            print(f"   📊 Description coverage: {len(description_annotation_ids)}/{len(annotation_ids)}")
            print(f"   📊 Motion coverage: {len(motion_annotation_ids)}/{len(annotation_ids)}")

            if missing_in_descriptions:
                print(f"   ⚠️ Missing in descriptions: {list(missing_in_descriptions)[:3]}...")
            if missing_in_motions:
                print(f"   ⚠️ Missing in motions: {list(missing_in_motions)[:3]}...")

            # Check point indices validity
            laser_scan_size = len(laser_scan.points)
            invalid_indices = []
            for ann in annotations[:3]:  # Check first few
                indices = ann.get('indices', [])
                if indices and max(indices) >= laser_scan_size:
                    invalid_indices.append(ann['annot_id'])

            if invalid_indices:
                print(f"   ⚠️ Invalid point indices in: {invalid_indices}")
            else:
                print(f"   ✅ Point indices are valid (checked sample)")

            consistency_score = 1.0 - (len(missing_in_descriptions) + len(missing_in_motions)) / (2 * len(annotation_ids))

            self.test_results["consistency"] = {
                "success": True,
                "annotation_coverage": len(description_annotation_ids) / len(annotation_ids),
                "motion_coverage": len(motion_annotation_ids) / len(annotation_ids),
                "consistency_score": consistency_score
            }

            return True

        except Exception as e:
            print(f"   ❌ Consistency test failed: {e}")
            self.test_results["consistency"] = {"success": False, "error": str(e)}
            return False

    def run_all_tests(self):
        """Run all validation tests."""
        print("🧪 Running Official SceneFun3D DataParser Validation")
        print("=" * 60)

        tests = [
            ("data_structure", self.validate_data_structure),
            ("laser_scan", self.test_laser_scan_loading),
            ("arkit_reconstruction", self.test_arkit_reconstruction_loading),
            ("transform", self.test_transformation_matrix),
            ("annotations", self.test_annotations_loading),
            ("descriptions", self.test_descriptions_loading),
            ("motions", self.test_motions_loading),
            ("consistency", self.test_data_consistency)
        ]

        passed_tests = 0
        total_tests = len(tests)

        for test_name, test_func in tests:
            success = test_func()
            if success:
                passed_tests += 1

        # Generate summary
        print(f"\n📊 VALIDATION SUMMARY")
        print("=" * 30)
        print(f"✅ Passed: {passed_tests}/{total_tests} tests")
        print(f"📈 Success rate: {passed_tests/total_tests*100:.1f}%")

        if passed_tests == total_tests:
            print("🎉 All tests passed! DataParser is ready for integration.")
        else:
            print("⚠️ Some tests failed. Review errors before integration.")

        return self.test_results

    def save_results(self, output_path: str = None):
        """Save test results to JSON file."""
        if output_path is None:
            output_path = f"dataparser_validation_{self.visit_id}_{self.video_id}.json"

        results_summary = {
            "test_config": {
                "data_root": str(self.data_root),
                "visit_id": self.visit_id,
                "video_id": self.video_id
            },
            "test_results": self.test_results,
            "summary": {
                "total_tests": len(self.test_results),
                "passed_tests": sum(1 for result in self.test_results.values()
                                  if isinstance(result, dict) and result.get("success", False)),
            }
        }

        with open(output_path, 'w') as f:
            json.dump(results_summary, f, indent=2)

        print(f"📝 Results saved to: {output_path}")
        return output_path


def main():
    """Main function to run DataParser validation."""
    parser = argparse.ArgumentParser(description="Validate SceneFun3D DataParser functionality")
    parser.add_argument("--data-root", required=True, help="Root directory containing SceneFun3D data")
    parser.add_argument("--visit-id", default="422203", help="Visit ID to test (default: 422203)")
    parser.add_argument("--video-id", default="42445781", help="Video ID to test (default: 42445781)")
    parser.add_argument("--output", help="Output file for test results (optional)")

    args = parser.parse_args()

    try:
        validator = OfficialParserValidator(args.data_root, args.visit_id, args.video_id)
        test_results = validator.run_all_tests()
        validator.save_results(args.output)

        # Exit with appropriate code
        success_count = sum(1 for result in test_results.values()
                          if isinstance(result, dict) and result.get("success", False))
        total_count = len(test_results)

        if success_count == total_count:
            print("\n🎯 DataParser validation completed successfully!")
            sys.exit(0)
        else:
            print(f"\n⚠️ DataParser validation completed with {total_count - success_count} failures.")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Fatal error during validation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()