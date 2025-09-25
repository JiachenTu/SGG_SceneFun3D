#!/usr/bin/env python3
"""
SceneFun3D Scene Graph Generation Pipeline

This pipeline transforms ARKitScenes objects to SceneFun3D coordinates
and generates hierarchical scene graphs with affordance relationships.
"""

import sys
import os
import numpy as np
import json
from pathlib import Path

# Add SceneFun3D to path
scenefun3d_root = Path("/home/jiachen/scratch/SceneFun3D/scenefun3d")
original_cwd = os.getcwd()
os.chdir(str(scenefun3d_root))
sys.path.insert(0, str(scenefun3d_root))

try:
    from utils.data_parser import DataParser
    print("✅ Successfully imported DataParser")
finally:
    os.chdir(original_cwd)

# Add local utils
sys.path.append(str(Path(__file__).parent / "utils"))
from arkitscenes_parser import ARKitScenesParser

def run_pipeline():
    """Run the scene graph generation pipeline."""
    print("🧪 SceneFun3D Scene Graph Generation Pipeline")
    print("=" * 50)

    # Configuration
    data_root = "/home/jiachen/scratch/SceneFun3D/alignment/data_examples/scenefun3d"
    arkitscenes_file = "/home/jiachen/scratch/SceneFun3D/alignment/data_examples/arkitscenes/video_42445781/42445781_3dod_annotation.json"
    visit_id = "422203"
    video_id = "42445781"

    try:
        # 1. Test SceneFun3D DataParser
        print("\n1️⃣ Testing SceneFun3D DataParser...")
        parser = DataParser(data_root)

        # Load core data
        laser_scan = parser.get_laser_scan(visit_id)
        transform_matrix = parser.get_transform(visit_id, video_id)
        annotations = parser.get_annotations(visit_id)
        descriptions = parser.get_descriptions(visit_id)
        motions = parser.get_motions(visit_id)

        print(f"   ✅ Laser scan: {len(laser_scan.points):,} points")
        print(f"   ✅ Transform matrix: {transform_matrix.shape}")
        print(f"   ✅ Annotations: {len(annotations)}")
        print(f"   ✅ Descriptions: {len(descriptions)}")
        print(f"   ✅ Motions: {len(motions)}")

        # Create mapping from affordance ID to task description
        affordance_task_map = {}
        for desc in descriptions:
            task_desc = desc.get('description', 'Unknown task')
            annot_ids = desc.get('annot_id', [])
            for annot_id in annot_ids:
                affordance_task_map[annot_id] = task_desc
        print(f"   📋 Task mapping created for {len(affordance_task_map)} affordances")

        # 2. Test ARKitScenes parser
        print("\n2️⃣ Testing ARKitScenes parser...")
        arkit_parser = ARKitScenesParser(arkitscenes_file)
        arkit_parser.load()
        objects = arkit_parser.get_objects()
        print(f"   ✅ ARKit objects: {len(objects)}")

        # 3. Test coordinate transformation
        print("\n3️⃣ Testing coordinate transformation...")
        inverse_transform = np.linalg.inv(transform_matrix)

        # Transform first object center (already in meters from parser)
        if objects:
            obj = objects[0]
            arkit_center_m = obj.obb.centroid  # Already in meters from updated parser
            print(f"   📊 ARKit center (m): {arkit_center_m}")

            # Transform to SceneFun3D coordinates
            homogeneous_point = np.append(arkit_center_m, 1)
            scenefun3d_center = (inverse_transform @ homogeneous_point)[:3]
            print(f"   📊 SceneFun3D center (m): {scenefun3d_center}")

        # 4. Test affordance extraction
        print("\n4️⃣ Testing affordance extraction...")
        laser_points = np.asarray(laser_scan.points)

        for i, desc in enumerate(descriptions[:2]):  # Test first 2 tasks
            task_name = desc.get('description', f'Task {i+1}')
            annot_ids = desc.get('annot_ids', [])
            print(f"   📋 {task_name}: {len(annot_ids)} annotation(s)")

            for annot_id in annot_ids[:1]:  # Test first annotation
                # Find annotation
                annotation = next((ann for ann in annotations if ann['annot_id'] == annot_id), None)
                if annotation:
                    indices = annotation.get('indices', [])
                    if indices:
                        affordance_points = laser_points[indices]
                        center = np.mean(affordance_points, axis=0)
                        print(f"      {annot_id}: {len(affordance_points)} points, center {center}")

        # 5. Test spatial relationships
        print("\n5️⃣ Testing spatial relationships...")

        if objects and annotations:
            obj = objects[0]  # First object
            annotation = annotations[0]  # First annotation

            # Transform object to SceneFun3D coordinates (already in meters)
            obj_center_m = obj.obb.centroid
            obj_center_scenefun3d = (inverse_transform @ np.append(obj_center_m, 1))[:3]

            # Get affordance points
            indices = annotation.get('indices', [])
            if indices:
                affordance_points = laser_points[indices]
                affordance_center = np.mean(affordance_points, axis=0)

                # Calculate distance
                distance = np.linalg.norm(obj_center_scenefun3d - affordance_center)
                print(f"   📏 Distance between {obj.label} and annotation: {distance:.2f}")

                # Simple confidence calculation
                confidence = max(0, 1 - distance / 500)  # Simple inverse distance
                print(f"   🎯 Simple confidence: {confidence:.3f}")

        # 6. Create simple scene graph
        print("\n6️⃣ Creating simple scene graph...")
        scene_graph = {
            "visit_id": visit_id,
            "video_id": video_id,
            "objects": [],
            "affordances": [],
            "relationships": []
        }

        # Add objects (coordinates already in meters from parser)
        for obj in objects:
            obj_center_m = obj.obb.centroid  # Already in meters
            obj_size_m = obj.obb.axes_lengths  # Already in meters
            obj_axes = obj.obb.get_rotation_matrix()  # Get 3x3 rotation matrix

            # Transform center to SceneFun3D coordinates
            obj_center_scenefun3d = (inverse_transform @ np.append(obj_center_m, 1))[:3]

            # Transform rotation axes to SceneFun3D coordinate frame
            obj_axes_scenefun3d = inverse_transform[:3, :3] @ obj_axes

            scene_graph["objects"].append({
                "id": obj.uid or f"object_{len(scene_graph['objects'])}",
                "class": obj.label,
                "center_arkit_m": obj_center_m.tolist(),
                "center_scenefun3d": obj_center_scenefun3d.tolist(),
                "size_m": obj_size_m.tolist(),
                "axes_arkit": obj_axes.tolist(),  # Original orientation in ARKit frame
                "axes_scenefun3d": obj_axes_scenefun3d.tolist()  # Transformed orientation in SceneFun3D frame
            })

        # Add affordances with 5mm voxel resolution handling
        VOXEL_SIZE = 0.005  # 5mm in meters

        def quantize_to_voxel_grid(size_array):
            """Round size to nearest 5mm increment to account for laser scan resolution."""
            return np.ceil(size_array / VOXEL_SIZE) * VOXEL_SIZE

        def enforce_minimum_size(size_array):
            """Enforce minimum affordance size based on voxel resolution."""
            min_affordance_size = np.array([VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE])
            return np.maximum(size_array, min_affordance_size)

        for i, annotation in enumerate(annotations[:3]):  # First 3 annotations
            indices = annotation.get('indices', [])
            if indices:
                affordance_points = laser_points[indices]
                center = np.mean(affordance_points, axis=0)

                # Calculate bounding box
                min_coords = np.min(affordance_points, axis=0)
                max_coords = np.max(affordance_points, axis=0)
                raw_size = max_coords - min_coords

                # Account for 5mm voxel resolution
                size_quantized = quantize_to_voxel_grid(raw_size)
                size = enforce_minimum_size(size_quantized)

                # Update min/max coords to match corrected size
                size_diff = size - raw_size
                min_coords -= size_diff / 2
                max_coords += size_diff / 2

                # Get task description for this affordance
                task_desc = affordance_task_map.get(annotation['annot_id'], 'Unknown task')

                print(f"   📍 Affordance {i+1} ({annotation['annot_id'][:8]}):")
                print(f"      Task: {task_desc}")
                print(f"      Points: {len(affordance_points)}")
                print(f"      Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}] m")
                print(f"      Raw size: [{raw_size[0]:.3f}, {raw_size[1]:.3f}, {raw_size[2]:.3f}] m")
                print(f"      Voxel-corrected size: [{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}] m")
                print(f"      Volume: {np.prod(size):.9f} m³")
                print(f"      Point density: {len(affordance_points)/np.prod(size):.0f} points/m³")

                scene_graph["affordances"].append({
                    "id": annotation['annot_id'],
                    "task_description": task_desc,
                    "center": center.tolist(),
                    "point_count": len(affordance_points),
                    "min_coords": min_coords.tolist(),
                    "max_coords": max_coords.tolist(),
                    "size": size.tolist(),
                    "raw_size": raw_size.tolist(),
                    "volume_m3": float(np.prod(size)),
                    "point_density_per_m3": float(len(affordance_points)/np.prod(size)),
                    "voxel_size_m": VOXEL_SIZE
                })

        # Calculate all object-affordance relationships
        print("\n7️⃣ Calculating spatial relationships...")
        for obj_data in scene_graph["objects"]:
            for aff_data in scene_graph["affordances"]:
                obj_center = np.array(obj_data["center_scenefun3d"])
                aff_center = np.array(aff_data["center"])

                distance = np.linalg.norm(obj_center - aff_center)
                # Use 1.0 meter threshold for reasonable confidence
                confidence = max(0, 1 - distance / 1.0)

                # Only store relationships with meaningful confidence (> 0.1)
                if confidence > 0.1:
                    scene_graph["relationships"].append({
                        "object_id": obj_data["id"],
                        "object_class": obj_data["class"],
                        "affordance_id": aff_data["id"],
                        "affordance_task": aff_data["task_description"],
                        "distance": distance,
                        "confidence": confidence
                    })
                    print(f"   🔗 {obj_data['class']} ↔ {aff_data['task_description']}: distance={distance:.2f}m, confidence={confidence:.3f}")

        print(f"   ✅ Scene graph created: {len(scene_graph['objects'])} objects, {len(scene_graph['affordances'])} affordances, {len(scene_graph['relationships'])} relationships")

        # 8. Save results
        output_file = "results/pipeline_results.json"
        os.makedirs("results", exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(scene_graph, f, indent=2)
        print(f"   💾 Results saved to: {output_file}")

        print("\n🎉 Pipeline completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)