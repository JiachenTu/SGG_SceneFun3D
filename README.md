# SceneFun3D Scene Graph Generation Pipeline

A pipeline for generating hierarchical scene graphs by integrating ARKitScenes 3D object detection with SceneFun3D affordance annotations.

## Features

- **Consistent Meter Units**: All coordinates and measurements in meters throughout the pipeline
- **5mm Voxel Resolution Awareness**: Properly handles SceneFun3D laser scan resolution limits
- **Coordinate Transformation**: Transform ARKit objects to SceneFun3D coordinate space
- **Official Integration**: Uses the official SceneFun3D DataParser toolkit
- **Floating Affordances**: Supports unattached affordances when spatial confidence is low
- **Rich Visualizations**: Generates comprehensive plots showing transformations and relationships
- **Simple Configuration**: Works with any data root path

## Installation

### Dependencies

```bash
pip install numpy matplotlib open3d
```

### SceneFun3D Toolkit

The pipeline requires the official SceneFun3D toolkit. Make sure it's installed at:
```
/home/jiachen/scratch/SceneFun3D/scenefun3d
```

## Quick Start

### 1. Run the Pipeline

```bash
python pipeline.py
```

This will:
- Load SceneFun3D laser scan data and annotations
- Parse ARKitScenes 3D object detections
- Transform objects to SceneFun3D coordinates
- Generate scene graph with spatial relationships
- Save results to `results/pipeline_results.json`

### 2. Generate Visualizations

```bash
python create_visualizations.py
```

This creates 4 visualization files in the `visualizations/` directory:
- `coordinate_comparison.png` - Before/after coordinate transformation
- `3d_scene_visualization.png` - 3D scene with objects and affordances
- `scene_graph_structure.png` - Hierarchical graph diagram
- `transformation_validation.png` - Accuracy validation plots

## Configuration

The pipeline uses example data by default. To use your own data, edit the paths in `pipeline.py`:

```python
# Configuration
data_root = "/path/to/your/scenefun3d/data"
arkitscenes_file = "/path/to/your/arkitscenes/annotation.json"
visit_id = "your_visit_id"
video_id = "your_video_id"
```

Or create a `config.yaml` file:

```yaml
data:
  scenefun3d_root: "/path/to/scenefun3d/data"
  arkitscenes_file: "/path/to/arkitscenes/annotation.json"
  visit_id: "422203"
  video_id: "42445781"
```

## Pipeline Components

### Core Modules

- `pipeline.py` - Main pipeline orchestration
- `utils/arkitscenes_parser.py` - ARKit data parsing
- `utils/unified_data_loader.py` - SceneFun3D data loading
- `utils/enhanced_coordinate_transform.py` - Coordinate transformations
- `utils/spatial_scorer.py` - Spatial confidence scoring
- `create_visualizations.py` - Visualization generation

### Data Structure

```
alignment/
├── README.md                    # This file
├── pipeline.py                  # Main pipeline
├── create_visualizations.py     # Visualization tool
├── config.yaml                  # Configuration template
├── utils/                       # Core utilities
├── data_examples/               # Example data
│   ├── arkitscenes/
│   └── scenefun3d/
├── results/                     # Pipeline outputs
│   └── pipeline_results.json
├── visualizations/              # Generated plots
└── archive/                     # Archived old files
```

## Output Format

The pipeline generates a JSON scene graph with this structure (all units in meters):

```json
{
  "visit_id": "422203",
  "video_id": "42445781",
  "objects": [
    {
      "id": "MhdKxnWEXdHg9Nle",
      "class": "bathtub",
      "center_arkit_m": [0.058, 0.241, 0.155],
      "center_scenefun3d": [-0.044, -0.434, 92.551],
      "size_m": [0.164, 0.061, 0.073]
    }
  ],
  "affordances": [
    {
      "id": "0682ad6f-09b5-4257-be3d-9a615bc9283a",
      "task_description": "Close the bathroom door",
      "center": [-0.342, 0.039, 92.930],
      "point_count": 186,
      "size": [0.040, 0.050, 0.055],
      "raw_size": [0.037, 0.048, 0.050],
      "volume_m3": 0.00011,
      "point_density_per_m3": 1690909,
      "voxel_size_m": 0.005
    }
  ],
  "relationships": [
    {
      "object_id": "MhdKxnWEXdHg9Nle",
      "object_class": "bathtub",
      "affordance_id": "0682ad6f-09b5-4257-be3d-9a615bc9283a",
      "affordance_task": "Close the bathroom door",
      "distance": 0.676,
      "confidence": 0.324
    }
  ]
}
```

## Technical Details

### Unit Handling and Resolution

**IMPORTANT - ARKitScenes Units Correction:**
- ⚠️ **ARKitScenes 3D annotations are in CENTIMETERS, not millimeters**
- Analysis shows realistic object dimensions with cm→m conversion:
  - Bathtub: [164.4, 60.6, 73.1] cm → [1.64, 0.61, 0.73] m ✅
  - Toilet: [42.0, 79.5, 67.2] cm → [0.42, 0.80, 0.67] m ✅
  - Sink: [56.9, 23.1, 43.7] cm → [0.57, 0.23, 0.44] m ✅
- Millimeter conversion produces unrealistically small objects (4-16 cm total size)

**Consistent Meter Units:**
- ARKitScenes data converted from **centimeters** to meters (÷100, not ÷1000)
- All coordinates, sizes, and distances in meters throughout pipeline
- Eliminates unit conversion errors and improves clarity

**5mm Voxel Resolution Handling:**
SceneFun3D laser scans are pre-processed with 5mm voxel downsampling:
- Combined from multiple Faro Focus S70 scanner positions
- Downsampled to 5mm resolution preserving functional details
- Pipeline accounts for this resolution limit in affordance size calculations

```python
VOXEL_SIZE = 0.005  # 5mm in meters

# Quantize affordance sizes to voxel grid
size_quantized = np.ceil(raw_size / VOXEL_SIZE) * VOXEL_SIZE

# Enforce minimum size (one voxel)
min_size = np.array([VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE])
size_corrected = np.maximum(size_quantized, min_size)
```

### Coordinate Transformation

The pipeline transforms ARKitScenes objects (already converted to meters) to SceneFun3D coordinate space:

```python
# Load transformation matrix
transform_matrix = parser.get_transform(visit_id, video_id)
inverse_transform = np.linalg.inv(transform_matrix)

# Transform ARKit point to SceneFun3D (both in meters)
homogeneous_point = np.append(arkit_center_m, 1)
scenefun3d_center = (inverse_transform @ homogeneous_point)[:3]
```

### Spatial Confidence Calculation

Confidence scores combine **overlap ratio** and **distance scoring** with **voxel resolution awareness**:

#### Formula:
```python
final_confidence = overlap_weight × overlap_ratio + distance_weight × distance_score
```

#### Components:

1. **Distance Score (40% weight):**
```python
# Account for 5mm voxel spatial uncertainty
spatial_uncertainty = √3 × (voxel_size / 2) ≈ 4.3mm
adjusted_distance = max(0, raw_distance - spatial_uncertainty)
normalized_distance = min(adjusted_distance / max_distance, 1.0)
distance_score = 1.0 - normalized_distance
```

2. **Overlap Ratio (60% weight):**
```python
# Calculate bounding box intersection
intersection_volume = calculate_bbox_intersection(object_bbox, affordance_bbox)
smaller_volume = min(object_volume, affordance_volume)
overlap_ratio = intersection_volume / smaller_volume
```

#### Key Parameters:
- **max_distance**: 0.5m (reasonable interaction range)
- **confidence_threshold**: 0.2 (lowered due to voxel sampling uncertainty)
- **spatial_uncertainty**: ±4.3mm (from 5mm voxel sampling)
- **voxel_size**: 0.005m (5mm laser scan resolution)

#### Confidence Interpretation:
- **> 0.7**: High confidence relationship
- **0.2-0.7**: Moderate confidence (above threshold)
- **< 0.2**: Low confidence → "floating" affordance
- **< 0.1**: Very low confidence → likely unrelated

#### No Fallback Logic:
Affordances remain "floating" (unattached) if no object relationship exceeds the confidence threshold. This prevents false associations and maintains data integrity.

### Example Results

**Bathroom Scene (visit_422203, video_42445781):**
- Objects: bathtub, sink, toilet (all coordinates in meters)
- Affordances: 3 regions with 186, 39, and 84 points (5mm voxel-corrected)
- Coordinate Transformations (ARKit → SceneFun3D):
  - Bathtub: [0.058, 0.241, 0.155]m → [-0.044, -0.434, 92.551]m
  - Sink: [0.082, 0.281, 0.067]m → [-0.005, -0.409, 92.464]m
  - Toilet: [0.029, 0.249, -0.010]m → [-0.066, -0.413, 92.386]m
- Affordance Sizes (voxel-corrected):
  - Door handle: raw [0.037, 0.048, 0.050]m → corrected [0.040, 0.050, 0.055]m
  - Point densities: 1.69M-3.36M points/m³ (realistic for 5mm voxels)
- Spatial Relationships:
  - Best confidence: 0.324 (bathtub ↔ door handle, 0.68m distance)
  - All affordances remain "floating" (below 0.2 threshold) - realistic for door handles

## Troubleshooting

**Import Error**: Make sure SceneFun3D toolkit is installed at the correct path
**File Not Found**: Check that example data exists in `data_examples/` directory
**Visualization Error**: Install matplotlib and ensure results file exists
**Unit Errors**: All coordinates should be in meters - check ARKitScenes parser output
**Low Confidence**: Confidence < 0.2 is normal for distant relationships; check voxel resolution settings

