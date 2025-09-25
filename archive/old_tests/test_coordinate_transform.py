#!/usr/bin/env python3
"""
Test Coordinate Transformation - V2.0 Pipeline

Comprehensive testing for the enhanced coordinate transformation utilities.
Tests both individual transformations and integration with data loaders.

Usage:
    python test_coordinate_transform.py --data-root /path/to/data
    python test_coordinate_transform.py --matrix-only  # Test with sample matrix
    python test_coordinate_transform.py --help
"""

import sys
import os
import argparse
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))

try:
    from enhanced_coordinate_transform import (
        EnhancedCoordinateTransformer,
        create_transformer_from_data,
        create_transformer_from_matrix,
        TransformationInfo
    )
    from unified_data_loader import UnifiedDataLoader, SceneData
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("   Make sure utils/enhanced_coordinate_transform.py and utils/unified_data_loader.py exist")
    sys.exit(1)


class CoordinateTransformTester:
    """Comprehensive tester for coordinate transformations."""

    def __init__(self, data_root: str = None, visit_id: str = "422203", video_id: str = "42445781"):
        """Initialize tester."""
        self.data_root = Path(data_root) if data_root else None
        self.visit_id = visit_id
        self.video_id = video_id
        self.test_results = {}

        print(f"🧪 Coordinate Transform Tester - V2.0")
        print(f"   📁 Data root: {self.data_root}")
        print(f"   🏠 Visit ID: {self.visit_id}")
        print(f"   🎥 Video ID: {self.video_id}")

    def test_matrix_only_transformation(self):
        """Test transformation with a known matrix."""
        print("\n1️⃣ Testing matrix-only transformation...")

        # Create a sample 4x4 transformation matrix
        # Simple rotation (45°) + translation
        angle = np.pi / 4  # 45 degrees
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        sample_matrix = np.array([
            [cos_a, -sin_a, 0, 100],
            [sin_a,  cos_a, 0, 200],
            [0,      0,     1, 50],
            [0,      0,     0, 1]
        ])

        try:
            transformer = create_transformer_from_matrix(sample_matrix)

            # Test basic transformation
            test_point = np.array([10, 20, 30])
            transformed = transformer.transform_arkit_to_scenefun3d(test_point)

            print(f"   ✅ Sample matrix transformation: {test_point} -> {transformed}")

            # Test round-trip accuracy
            accuracy = transformer.validate_round_trip_accuracy()
            print(f"   ✅ Round-trip accuracy: {accuracy['is_accurate']} (error: {accuracy['max_error']:.8f})")

            # Test transformation info
            info = transformer.get_transformation_info()
            print(f"   ✅ Transform info: rotation={info.rotation_angle_deg:.2f}°, valid={info.is_valid}")

            self.test_results["matrix_only"] = {
                "success": True,
                "round_trip_accuracy": accuracy,
                "transform_info": {
                    "rotation_angle": info.rotation_angle_deg,
                    "is_valid": info.is_valid,
                    "determinant": info.determinant
                }
            }
            return True

        except Exception as e:
            print(f"   ❌ Matrix-only test failed: {e}")
            self.test_results["matrix_only"] = {"success": False, "error": str(e)}
            return False

    def test_data_loader_integration(self):
        """Test integration with UnifiedDataLoader."""
        if not self.data_root or not self.data_root.exists():
            print("\n2️⃣ Skipping data loader integration (no data root)")
            return True

        print("\n2️⃣ Testing data loader integration...")

        try:
            # Initialize transformer from data
            transformer = create_transformer_from_data(str(self.data_root), self.visit_id, self.video_id)

            # Test with UnifiedDataLoader
            loader = UnifiedDataLoader(str(self.data_root), self.visit_id, self.video_id)
            scene_data = loader.load_all_data(include_mesh=False, include_point_cloud=False)

            print(f"   ✅ Scene data loaded: {len(scene_data.load_errors)} errors")
            print(f"   ✅ Transform matrix shape: {scene_data.transform_matrix.shape}")

            # Create transformer from loaded matrix
            transformer2 = create_transformer_from_matrix(scene_data.transform_matrix)

            # Compare transformers
            test_points = np.random.uniform(-100, 100, (5, 3))
            result1 = transformer.transform_arkit_to_scenefun3d(test_points)
            result2 = transformer2.transform_arkit_to_scenefun3d(test_points)

            difference = np.max(np.abs(result1 - result2))
            print(f"   ✅ Transformer consistency: {difference:.8f} max difference")

            self.test_results["data_integration"] = {
                "success": True,
                "scene_data_errors": len(scene_data.load_errors),
                "transformer_consistency": difference < 1e-10
            }
            return True

        except Exception as e:
            print(f"   ❌ Data loader integration test failed: {e}")
            self.test_results["data_integration"] = {"success": False, "error": str(e)}
            return False

    def test_batch_transformation(self):
        """Test batch transformation of objects."""
        print("\n3️⃣ Testing batch transformation...")

        try:
            # Create transformer (use sample matrix if no data)
            if self.data_root and self.data_root.exists():
                transformer = create_transformer_from_data(str(self.data_root), self.visit_id, self.video_id)
            else:
                # Use identity matrix for testing
                transformer = create_transformer_from_matrix(np.eye(4))

            # Create test objects
            test_objects = [
                {
                    'id': 'toilet',
                    'semantic_class': 'toilet',
                    'center': [29.01, 249.20, -9.67],
                    'size': [41.99, 79.52, 67.17],
                    'rotation': np.eye(3)
                },
                {
                    'id': 'sink',
                    'semantic_class': 'sink',
                    'center': [81.51, 280.84, 67.49],
                    'size': [56.94, 23.10, 43.70],
                    'rotation': np.eye(3)
                },
                {
                    'id': 'bathtub',
                    'semantic_class': 'bathtub',
                    'center': [57.75, 241.14, 154.95],
                    'size': [164.39, 60.64, 73.11],
                    'rotation': np.eye(3)
                }
            ]

            # Batch transform
            transformed_objects = transformer.batch_transform_objects(test_objects)

            print(f"   ✅ Batch transformation: {len(transformed_objects)} objects")

            success_count = 0
            for obj in transformed_objects:
                if 'transformation_error' not in obj:
                    success_count += 1
                    print(f"      {obj['id']}: {obj['center'][:3]}")
                else:
                    print(f"      {obj['id']}: ERROR - {obj['transformation_error']}")

            self.test_results["batch_transformation"] = {
                "success": True,
                "objects_processed": len(transformed_objects),
                "success_count": success_count
            }

            return success_count == len(transformed_objects)

        except Exception as e:
            print(f"   ❌ Batch transformation test failed: {e}")
            self.test_results["batch_transformation"] = {"success": False, "error": str(e)}
            return False

    def test_bounding_box_transformation(self):
        """Test oriented bounding box transformation."""
        print("\n4️⃣ Testing bounding box transformation...")

        try:
            # Create transformer
            if self.data_root and self.data_root.exists():
                transformer = create_transformer_from_data(str(self.data_root), self.visit_id, self.video_id)
            else:
                # Create rotation matrix (30° around Z-axis)
                angle = np.pi / 6
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                transform = np.array([
                    [cos_a, -sin_a, 0, 10],
                    [sin_a,  cos_a, 0, 20],
                    [0,      0,     1, 5],
                    [0,      0,     0, 1]
                ])
                transformer = create_transformer_from_matrix(transform)

            # Test bounding box transformation
            center = np.array([100, 200, 50])
            size = np.array([10, 20, 5])
            rotation = np.array([
                [0.866, -0.5, 0],
                [0.5, 0.866, 0],
                [0, 0, 1]
            ])

            t_center, t_size, t_rotation = transformer.transform_oriented_bounding_box(
                center, size, rotation
            )

            print(f"   ✅ OBB transformation:")
            print(f"      Center: {center} -> {t_center}")
            print(f"      Size: {size} -> {t_size}")
            print(f"      Rotation shape: {rotation.shape} -> {t_rotation.shape}")

            # Test axis-aligned bounds
            min_bounds, max_bounds = transformer.get_axis_aligned_bbox_from_obb(
                t_center, t_size, t_rotation
            )

            print(f"   ✅ Axis-aligned bounds: {min_bounds} to {max_bounds}")

            self.test_results["bounding_box"] = {
                "success": True,
                "center_transformed": t_center.tolist(),
                "size_preserved": np.allclose(size, t_size),
                "bounds_valid": np.all(max_bounds >= min_bounds)
            }

            return True

        except Exception as e:
            print(f"   ❌ Bounding box test failed: {e}")
            self.test_results["bounding_box"] = {"success": False, "error": str(e)}
            return False

    def test_transformation_properties(self):
        """Test mathematical properties of transformations."""
        print("\n5️⃣ Testing transformation properties...")

        try:
            # Create transformer
            if self.data_root and self.data_root.exists():
                transformer = create_transformer_from_data(str(self.data_root), self.visit_id, self.video_id)
            else:
                # Create a rigid transformation
                transform = np.array([
                    [0.866, -0.5, 0, 100],
                    [0.5, 0.866, 0, 200],
                    [0, 0, 1, 50],
                    [0, 0, 0, 1]
                ])
                transformer = create_transformer_from_matrix(transform)

            # Test properties
            info = transformer.get_transformation_info()

            # Test linearity (transformation of linear combination)
            p1 = np.array([10, 20, 30])
            p2 = np.array([40, 50, 60])
            a, b = 0.3, 0.7

            # Transform combination
            combined = a * p1 + b * p2
            t_combined = transformer.transform_arkit_to_scenefun3d(combined)

            # Combination of transforms
            t_p1 = transformer.transform_arkit_to_scenefun3d(p1)
            t_p2 = transformer.transform_arkit_to_scenefun3d(p2)
            combination_of_transforms = a * t_p1 + b * t_p2

            linearity_error = np.linalg.norm(t_combined - combination_of_transforms)
            print(f"   ✅ Linearity test: error = {linearity_error:.8f}")

            # Test distance preservation (for rigid transformations)
            points = np.random.uniform(-100, 100, (10, 3))
            transformed_points = transformer.transform_arkit_to_scenefun3d(points)

            original_distances = []
            transformed_distances = []

            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    orig_dist = np.linalg.norm(points[i] - points[j])
                    trans_dist = np.linalg.norm(transformed_points[i] - transformed_points[j])
                    original_distances.append(orig_dist)
                    transformed_distances.append(trans_dist)

            distance_errors = np.abs(np.array(original_distances) - np.array(transformed_distances))
            max_distance_error = np.max(distance_errors)
            print(f"   ✅ Distance preservation: max error = {max_distance_error:.8f}")

            self.test_results["properties"] = {
                "success": True,
                "linearity_error": linearity_error,
                "distance_preservation_error": max_distance_error,
                "is_rigid": max_distance_error < 1e-6,
                "transform_info": {
                    "determinant": info.determinant,
                    "is_valid": info.is_valid
                }
            }

            return True

        except Exception as e:
            print(f"   ❌ Transformation properties test failed: {e}")
            self.test_results["properties"] = {"success": False, "error": str(e)}
            return False

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("\n6️⃣ Testing edge cases...")

        try:
            # Test with identity matrix
            identity_transformer = create_transformer_from_matrix(np.eye(4))

            # Test empty points
            empty_result = identity_transformer.transform_arkit_to_scenefun3d(np.array([]).reshape(0, 3))
            print(f"   ✅ Empty points: shape {empty_result.shape}")

            # Test single point
            single_point = np.array([1, 2, 3])
            single_result = identity_transformer.transform_arkit_to_scenefun3d(single_point)
            print(f"   ✅ Single point: {single_point} -> {single_result}")

            # Test large batch
            large_batch = np.random.uniform(-1000, 1000, (1000, 3))
            large_result = identity_transformer.transform_arkit_to_scenefun3d(large_batch)
            print(f"   ✅ Large batch: {large_batch.shape} -> {large_result.shape}")

            # Test invalid matrix (should fail gracefully)
            try:
                singular_matrix = np.zeros((4, 4))
                singular_matrix[3, 3] = 1  # Make it homogeneous but singular
                invalid_transformer = create_transformer_from_matrix(singular_matrix)
                print(f"   ❌ Singular matrix should have failed")
            except ValueError:
                print(f"   ✅ Singular matrix properly rejected")

            self.test_results["edge_cases"] = {
                "success": True,
                "empty_points": empty_result.shape[0] == 0,
                "single_point_works": np.allclose(single_point, single_result),  # Identity
                "large_batch_works": large_result.shape == large_batch.shape,
                "invalid_matrix_rejected": True
            }

            return True

        except Exception as e:
            print(f"   ❌ Edge cases test failed: {e}")
            self.test_results["edge_cases"] = {"success": False, "error": str(e)}
            return False

    def run_all_tests(self):
        """Run all coordinate transformation tests."""
        print("🧪 Running Coordinate Transformation Tests - V2.0")
        print("=" * 60)

        tests = [
            ("matrix_only", self.test_matrix_only_transformation),
            ("data_integration", self.test_data_loader_integration),
            ("batch_transformation", self.test_batch_transformation),
            ("bounding_box", self.test_bounding_box_transformation),
            ("properties", self.test_transformation_properties),
            ("edge_cases", self.test_edge_cases)
        ]

        passed_tests = 0
        total_tests = len(tests)

        for test_name, test_func in tests:
            success = test_func()
            if success:
                passed_tests += 1

        # Generate summary
        print(f"\n📊 TEST SUMMARY")
        print("=" * 30)
        print(f"✅ Passed: {passed_tests}/{total_tests} tests")
        print(f"📈 Success rate: {passed_tests/total_tests*100:.1f}%")

        if passed_tests == total_tests:
            print("🎉 All tests passed! Coordinate transformation is ready.")
        else:
            print("⚠️ Some tests failed. Review errors before proceeding.")

        return self.test_results

    def save_results(self, output_path: str = None):
        """Save test results to JSON file."""
        if output_path is None:
            output_path = f"coordinate_transform_test_results.json"

        results_summary = {
            "test_config": {
                "data_root": str(self.data_root) if self.data_root else None,
                "visit_id": self.visit_id,
                "video_id": self.video_id
            },
            "test_results": self._serialize_test_results(self.test_results),
            "summary": {
                "total_tests": len(self.test_results),
                "passed_tests": sum(1 for result in self.test_results.values()
                                  if isinstance(result, dict) and result.get("success", False)),
            }
        }

    def _serialize_test_results(self, results):
        """Convert test results to JSON-serializable format."""
        serialized = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serialized[key] = {}
                for k, v in value.items():
                    if isinstance(v, (bool, int, float, str, list)):
                        serialized[key][k] = v
                    elif isinstance(v, np.ndarray):
                        serialized[key][k] = v.tolist()
                    else:
                        serialized[key][k] = str(v)
            else:
                serialized[key] = str(value)
        return serialized

        with open(output_path, 'w') as f:
            json.dump(results_summary, f, indent=2)

        print(f"📝 Results saved to: {output_path}")
        return output_path


def main():
    """Main function to run coordinate transformation tests."""
    parser = argparse.ArgumentParser(description="Test coordinate transformation utilities")
    parser.add_argument("--data-root", help="Root directory containing SceneFun3D data")
    parser.add_argument("--visit-id", default="422203", help="Visit ID to test (default: 422203)")
    parser.add_argument("--video-id", default="42445781", help="Video ID to test (default: 42445781)")
    parser.add_argument("--matrix-only", action="store_true", help="Test only with sample matrices")
    parser.add_argument("--output", help="Output file for test results (optional)")

    args = parser.parse_args()

    try:
        # If matrix-only, don't use data root
        data_root = None if args.matrix_only else args.data_root

        tester = CoordinateTransformTester(data_root, args.visit_id, args.video_id)
        test_results = tester.run_all_tests()
        tester.save_results(args.output)

        # Exit with appropriate code
        success_count = sum(1 for result in test_results.values()
                          if isinstance(result, dict) and result.get("success", False))
        total_count = len(test_results)

        if success_count == total_count:
            print("\n🎯 Coordinate transformation tests completed successfully!")
            sys.exit(0)
        else:
            print(f"\n⚠️ Tests completed with {total_count - success_count} failures.")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Fatal error during testing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()