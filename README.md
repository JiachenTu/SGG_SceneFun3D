# ARKitScenes & SceneFun3D Data Alignment

This directory contains example data and alignment tools for integrating ARKitScenes 3D object detection with SceneFun3D affordance annotations.

## Example Data: Bathroom Scene

### Overview
- **Visit ID**: 422203 (SceneFun3D)
- **Video ID**: 42445781 (ARKitScenes)
- **Scene Type**: Bathroom with toilet, sink, bathtub
- **Task Descriptions**: 6 functional tasks

### Data Location

```
data_examples/
├── arkitscenes/
│   └── video_42445781/
│       ├── 42445781_3dod_annotation.json    # 3D object bounding boxes
│       ├── 42445781_3dod_mesh.ply          # ARKit mesh
│       └── 42445781_frames/                # RGB, depth, poses
└── scenefun3d/
    └── visit_422203/
        ├── 422203_annotations.json         # Affordance point indices
        ├── 422203_descriptions.json        # Task descriptions
        ├── 422203_motions.json            # Motion parameters
        ├── 422203_laser_scan.ply          # High-res laser scan
        └── 42445781/                       # Video-specific data
            ├── 42445781_transform.npy      # Coordinate transformation
            ├── 42445781_arkit_mesh.ply     # ARKit mesh
            ├── lowres_wide/                # RGB frames
            ├── lowres_depth/               # Depth maps
            └── lowres_poses.traj           # Camera poses
```

## ARKitScenes Data Analysis

### 3D Objects Detected
From `42445781_3dod_annotation.json`:

1. **Bathtub**
   - Center: [57.75, 241.14, 154.95] mm
   - Size: [164.39, 60.64, 73.11] mm
   - Real dimensions: ~164cm × 61cm × 73cm

2. **Sink**
   - Center: [81.51, 280.84, 67.49] mm
   - Size: [56.94, 23.10, 43.70] mm
   - Real dimensions: ~57cm × 23cm × 44cm

3. **Toilet**
   - Center: [29.01, 249.20, -9.67] mm
   - Size: [41.99, 79.52, 67.17] mm
   - Real dimensions: ~42cm × 80cm × 67cm

### Coordinate System
- Units: millimeters
- Origin: ARKitScenes coordinate system
- Rotation: Oriented bounding boxes with rotation matrices

## SceneFun3D Data Analysis

### Task Descriptions (6 tasks)
From `422203_descriptions.json`:

1. **"Close the bathroom door"**
   - Annotation IDs: 0682ad6f, e93e1b5c
   - Motion: Rotation around Z-axis

2. **"Flush the toilet"**
   - Annotation ID: bfc23a3d
   - Motion: Translation (push down)
   - Direction: [0, 0, -1]

3. **"Open the window above the sink"**
   - Annotation ID: 2777270f
   - Motion: Rotation
   - Direction: [0.999, -0.006, -0.014]

4. **"Open the window above the toilet"**
   - Annotation ID: 3e9e30c0
   - Motion: Rotation
   - Direction: [0.999, -0.006, -0.014]

5. **"Turn on the tap in the sink"**
   - Annotation ID: 90efb7a6
   - Motion: Rotation around Z-axis
   - Direction: [0, 0, -1]

6. **"Unplug the make up mirror"**
   - Annotation ID: 5866dc36
   - Motion: Translation (pull out)
   - Direction: [0.999, 0.038, 0.009]

### Affordance Types
- **Rotational**: Door handles, taps, windows
- **Translational**: Flush button, mirror plug

### Coordinate System
- Point indices into laser scan
- Transform matrix: `42445781_transform.npy`
- Motion origins specified by point indices

## Data Integration Challenges

### 1. Coordinate System Alignment
- **ARKitScenes**: 3D coordinates in millimeters
- **SceneFun3D**: Point indices into laser scan
- **Solution**: Use `42445781_transform.npy` to align coordinate systems

### 2. Scale and Units
- **ARKitScenes**: Metric measurements (mm)
- **SceneFun3D**: Point cloud indices
- **Solution**: Convert indices to 3D coordinates using laser scan

### 3. Object-Affordance Mapping
- **Challenge**: Link affordance regions to 3D objects
- **Approach**: Spatial overlap analysis between point regions and bounding boxes

### 4. Motion Parameter Translation
- **Challenge**: Convert point indices to 3D motion parameters
- **Approach**: Lookup 3D coordinates and compute motion vectors

## Analysis Scripts

### Planned Development

1. **`scripts/data_explorer.py`**
   - Parse both datasets
   - Visualize 3D objects and affordance regions
   - Apply coordinate transformations

2. **`scripts/scene_graph_builder.py`**
   - Build integrated scene graphs
   - One graph per task description
   - Combine objects and affordances

3. **`scripts/alignment_validator.py`**
   - Validate coordinate alignment
   - Check spatial relationships
   - Generate alignment metrics

4. **`utils/` modules**
   - ARKitScenes parser
   - SceneFun3D parser
   - Coordinate transformation utilities

## Expected Output

### Scene Graph Example: "Flush the toilet"

```python
{
    "visit_id": "422203",
    "video_id": "42445781",
    "description": "Flush the toilet",
    "object_nodes": [
        {
            "ID": "toilet_01",
            "semantic_class": "toilet",
            "3d_center": [29.01, 249.20, -9.67],
            "3d_bbox": {
                "size": [41.99, 79.52, 67.17],
                "rotation": [...]
            }
        }
    ],
    "affordance_nodes": [
        {
            "ID": "flush_button_01",
            "affordance_type": "Push",
            "motion_type": "trans",
            "motion_direction": [0, 0, -1],
            "attach_to": "toilet_01"
        }
    ]
}
```

## Implementation

### Utilities (`utils/`)

1. **`arkitscenes_parser.py`** - Parse ARKitScenes 3DOD annotations
   - Extract 3D bounding boxes, semantic classes, rotation matrices
   - Compute object volumes and spatial properties

2. **`scenefun3d_parser.py`** - Parse SceneFun3D data
   - Load task descriptions, affordance annotations, motion parameters
   - Infer affordance types from natural language

3. **`coordinate_transform.py`** - Handle coordinate system alignment
   - Apply transformation matrices between coordinate systems
   - Convert point indices to 3D coordinates

4. **`point_cloud_utils.py`** - Process laser scan point clouds
   - Load PLY files, extract point subsets, compute statistics
   - Handle large-scale point cloud operations

### Analysis Scripts (`scripts/`)

1. **`spatial_analyzer.py`** - Analyze object-affordance spatial relationships
   - Compute overlap ratios and distances
   - Find parent objects for affordances
   - Generate confidence scores

2. **`hierarchical_graph_builder.py`** - Build hierarchical 3D scene graphs
   - 4-level hierarchy: Scene → Region → Object → Affordance
   - One graph per task description
   - Include motion parameters and spatial reasoning

3. **`comprehensive_analysis.py`** - Complete analysis pipeline
   - Demonstrates full workflow from data loading to scene graph generation
   - Exports results to JSON files

## Pipeline Usage

### Quick Start
```bash
cd /home/jiachen/scratch/SceneFun3D/alignment
source /home/jiachen/miniconda3/etc/profile.d/conda.sh
conda activate scenefun3d

# Run setup (first time only)
bash setup.sh

# Run complete integrated pipeline
python run_pipeline.py --validate --verbose

# Validate outputs separately
python validate_outputs.py --verbose
```

### Pipeline Commands

#### 1. Integrated Pipeline Execution
```bash
# Basic execution
python run_pipeline.py

# With validation and verbose output (recommended)
python run_pipeline.py --validate --verbose

# Custom output directory
python run_pipeline.py --output-dir custom_outputs --validate
```

#### 2. Individual Component Testing
```bash
# Test individual parsers
python utils/arkitscenes_parser.py
python utils/scenefun3d_parser.py

# Test spatial analysis
python scripts/spatial_analyzer.py

# Test scene graph building
python scripts/hierarchical_graph_builder.py
```

#### 3. Output Validation
```bash
# Validate pipeline outputs
python validate_outputs.py

# Validate custom output directory
python validate_outputs.py --output-dir custom_outputs --verbose
```

#### 4. Scene Graph Visualization
```bash
# Visualize single scene graph
python scripts/scene_graph_visualizer.py --input outputs/scene_graphs/bathroom_422203_task_2_flush_the_toilet.json

# Generate all individual visualizations
python scripts/scene_graph_visualizer.py --input-dir outputs/scene_graphs/ --format svg

# Create batch comparison
python scripts/scene_graph_visualizer.py --input-dir outputs/scene_graphs/ --batch
```

## Pipeline Components

### Core Utilities (`utils/`)

#### ARKitScenes Parser (`arkitscenes_parser.py`)
- **Purpose**: Parse ARKitScenes 3D object detection annotations
- **Input**: `42445781_3dod_annotation.json`
- **Output**: `ARKitObject` instances with oriented bounding boxes, labels, and poses
- **Key Features**:
  - Oriented bounding box parsing with rotation matrices
  - Volume computation and spatial properties
  - Support for all ARKitScenes object classes

#### SceneFun3D Parser (`scenefun3d_parser.py`)
- **Purpose**: Parse SceneFun3D affordance annotations and task descriptions
- **Input**: Visit directory with annotations, descriptions, and motions JSON files
- **Output**: `TaskDescription` and `Annotation` instances with motion parameters
- **Key Features**:
  - Natural language task description parsing
  - Point cloud index extraction for affordance regions
  - Motion parameter extraction (type, direction, origin)
  - Automatic affordance type inference from descriptions

#### Coordinate Transformer (`coordinate_transform.py`)
- **Purpose**: Transform coordinates between SceneFun3D and ARKitScenes coordinate systems
- **Input**: `42445781_transform.npy` transformation matrix
- **Output**: Aligned 3D coordinates
- **Key Features**:
  - 4x4 homogeneous transformation matrix application
  - Batch point transformation for efficiency
  - Rotation angle extraction and validation

#### Point Cloud Processor (`point_cloud_utils.py`)
- **Purpose**: Process laser scan point clouds and extract affordance regions
- **Input**: `422203_laser_scan.ply` with 2.3M+ points
- **Output**: Point subsets for annotations and spatial statistics
- **Key Features**:
  - Efficient PLY file loading with Open3D
  - Point subset extraction by indices
  - Bounding box computation for point regions

### Analysis Scripts (`scripts/`)

#### Spatial Analyzer (`spatial_analyzer.py`)
- **Purpose**: Analyze spatial relationships between objects and affordances
- **Process**:
  1. Transform affordance points to ARKitScenes coordinates
  2. Compute overlap ratios with object bounding boxes
  3. Calculate confidence scores based on spatial proximity
- **Output**: `SpatialRelationship` instances with confidence metrics
- **Key Metrics**:
  - Overlap ratio: Percentage of affordance points inside object bbox
  - Distance: Euclidean distance between centers
  - Confidence: Combined score (overlap + proximity)

#### Hierarchical Graph Builder (`hierarchical_graph_builder.py`)
- **Purpose**: Build 4-level hierarchical scene graphs combining objects and affordances
- **Architecture**:
  - **Level 0**: Scene root (entire bathroom scene)
  - **Level 1**: Spatial regions (toilet_area, sink_area, bathtub_area)
  - **Level 2**: Objects (toilet, sink, bathtub from ARKitScenes)
  - **Level 3**: Affordances (flush, turn_tap, etc. from SceneFun3D)
- **Output**: `TaskSceneGraph` instances with spatial reasoning chains
- **Key Features**:
  - Automatic region assignment based on object proximity
  - Parent-child relationship establishment
  - Target affordance identification for each task
  - Spatial reasoning chain generation

#### Scene Graph Visualizer (`scene_graph_visualizer.py`)
- **Purpose**: Generate clear, hierarchical visualizations of scene graphs
- **Features**:
  - Color-coded nodes by type (scene/region/object/affordance)
  - Detailed multi-line labels with key attributes
  - Hierarchical tree layout with parent-child relationships
  - Multiple output formats (PNG, SVG, PDF)
- **Modes**:
  - Single graph visualization with detailed labeling
  - Batch comparison with grid layout
  - High-quality outputs for publications
- **Key Benefits**:
  - Intuitive visual understanding of scene structure
  - Easy comparison between different task graphs
  - Professional-quality diagrams for presentations

### Main Pipeline (`run_pipeline.py`)

#### Integrated Execution Pipeline
- **Purpose**: Orchestrate complete scene graph generation workflow
- **Process**:
  1. **Data Loading**: Load and validate all input files
  2. **Spatial Analysis**: Find object-affordance spatial relationships
  3. **Scene Graph Generation**: Build hierarchical graphs for each task
  4. **Output Generation**: Save JSON files and summary reports
  5. **Validation**: Run quality checks on outputs
- **Key Features**:
  - Comprehensive error handling and logging
  - Progress tracking with timestamps
  - Automatic output directory management
  - Built-in validation with detailed reporting

### Validation System (`validate_outputs.py`)

#### Output Quality Assurance
- **Purpose**: Validate completeness and correctness of pipeline outputs
- **Checks**:
  - Directory structure validation
  - Scene graph file completeness and format validation
  - Spatial analysis output verification
  - Data consistency between inputs and outputs
- **Output**: Detailed validation reports with success/failure metrics
- **Key Features**:
  - JSON schema validation for scene graphs
  - Statistical analysis of generated data
  - Confidence threshold verification
  - Comprehensive error reporting

### Output Structure
```
outputs/
├── scene_graphs/                   # Generated scene graphs
│   ├── bathroom_422203_task_1_close_the_bathroom_door.json
│   ├── bathroom_422203_task_2_flush_the_toilet.json
│   ├── bathroom_422203_task_3_open_the_window_above_the_sink.json
│   ├── bathroom_422203_task_4_open_the_window_above_the_toilet.json
│   ├── bathroom_422203_task_5_turn_on_the_tap_in_the_sink.json
│   └── bathroom_422203_task_6_unplug_the_make_up_mirror.json
├── visualizations/                 # Scene graph visualizations
│   ├── bathroom_422203_task_1_close_the_bathroom_door_graph.png
│   ├── bathroom_422203_task_2_flush_the_toilet_graph.png
│   ├── bathroom_422203_task_3_open_the_window_above_the_sink_graph.png
│   ├── bathroom_422203_task_4_open_the_window_above_the_toilet_graph.png
│   ├── bathroom_422203_task_5_turn_on_the_tap_in_the_sink_graph.png
│   ├── bathroom_422203_task_6_unplug_the_make_up_mirror_graph.png
│   └── scene_graphs_comparison.png
├── spatial_analysis/               # Spatial relationship data
│   └── spatial_relationships.json
├── validation/                     # Validation reports
│   └── validation_report_YYYYMMDD_HHMMSS.json
├── logs/                          # Execution logs
│   └── pipeline_execution_YYYYMMDD_HHMMSS.log
└── pipeline_summary.json          # Overall statistics
```

### Pipeline Features

#### 🔄 Integrated Execution
- Single-command pipeline execution with built-in visualization
- Automatic dependency checking
- Comprehensive error handling
- Progress reporting with timestamps

#### ✅ Built-in Validation
- Input file validation
- Output completeness checking
- Data consistency verification
- Quality metrics computation

#### 📊 Comprehensive Outputs
- Individual scene graphs per task
- Hierarchical visualizations with color-coded nodes
- Spatial relationship analysis
- Execution logs and summaries
- Validation reports

#### 🛠️ Robust Error Handling
- Graceful failure handling
- Clear error messages
- Partial execution with warnings
- Recovery mechanisms

### Scene Graph Structure
```python
{
  "task_description": "Flush the toilet",
  "nodes": {
    "scene_root": {...},
    "toilet_area": {...},
    "toilet_object": {
      "semantic_class": "toilet",
      "bbox_center": [29.01, 249.20, -9.67],
      "bbox_size": [41.99, 79.52, 67.17]
    },
    "flush_affordance": {
      "affordance_type": "Push",
      "motion_type": "trans",
      "motion_direction": [0, 0, -1],
      "confidence": 0.85
    }
  },
  "target_affordances": ["flush_affordance"],
  "spatial_reasoning_chain": [
    "Locate toilet in the scene",
    "Identify push affordance on toilet",
    "Execute translational motion"
  ]
}
```

## Scene Graph Visualization

The pipeline includes a comprehensive visualization system that generates clear, hierarchical diagrams of scene graphs with detailed node labeling and color coding.

### Visualization Features

- **Hierarchical Layout**: 4-level tree structure (Scene → Region → Object → Affordance)
- **Color-Coded Nodes**: Different colors for each node type
  - 🔵 **Scene Root**: Deep blue - entire scene overview
  - 🟢 **Spatial Regions**: Green - toilet_area, sink_area, bathtub_area
  - 🟠 **Objects**: Orange - toilet, sink, bathtub with properties
  - 🔴 **Affordances**: Red - interactive elements with motion parameters
- **Detailed Labels**: Multi-line labels showing key attributes for each node
- **Multiple Formats**: PNG, SVG, PDF output for publications and presentations

### Usage Commands

#### Single Scene Graph Visualization
```bash
# Visualize specific task
python scripts/scene_graph_visualizer.py --input outputs/scene_graphs/bathroom_422203_task_2_flush_the_toilet.json

# Custom output directory and format
python scripts/scene_graph_visualizer.py --input outputs/scene_graphs/bathroom_422203_task_2_flush_the_toilet.json --output-dir my_visualizations --format svg
```

#### Batch Visualization
```bash
# Generate individual visualizations for all scene graphs
python scripts/scene_graph_visualizer.py --input-dir outputs/scene_graphs/ --output-dir visualizations/

# Create comparison grid of all scene graphs
python scripts/scene_graph_visualizer.py --input-dir outputs/scene_graphs/ --batch --output-dir visualizations/

# Verbose output with processing details
python scripts/scene_graph_visualizer.py --input-dir outputs/scene_graphs/ --batch --verbose
```

### Output Files

**Individual Visualizations:**
```
outputs/visualizations/
├── bathroom_422203_task_1_close_the_bathroom_door_graph.png
├── bathroom_422203_task_2_flush_the_toilet_graph.png
├── bathroom_422203_task_3_open_the_window_above_the_sink_graph.png
├── bathroom_422203_task_4_open_the_window_above_the_toilet_graph.png
├── bathroom_422203_task_5_turn_on_the_tap_in_the_sink_graph.png
└── bathroom_422203_task_6_unplug_the_make_up_mirror_graph.png
```

**Batch Comparison:**
```
outputs/visualizations/
└── scene_graphs_comparison.png
```

**Automatic Generation:** Visualizations are automatically generated during pipeline execution and saved alongside scene graphs in the `outputs/` directory structure.

### Node Label Content

Each node type displays specific information:

**Scene Root Nodes:**
- Scene description (e.g., "Bathroom scene")
- Visit ID and Video ID
- Spatial bounds summary

**Spatial Region Nodes:**
- Region name (toilet_area, sink_area, bathtub_area)
- Primary objects in the region
- Region size information

**Object Nodes:**
- Semantic class (toilet, sink, bathtub)
- Volume in mm³
- Occlusion and orientation attributes

**Affordance Nodes:**
- Affordance type (Push, Rotate, Pull, Interact)
- Motion type and direction
- Confidence score (0.0-1.0)
- Point count from annotation

### Visualization Options

#### Command Line Arguments
- `--input`: Single JSON file path
- `--input-dir`: Directory containing multiple JSON files
- `--output-dir`: Output directory (default: visualizations)
- `--format`: Output format - png, svg, pdf (default: png)
- `--batch`: Create grid comparison of multiple graphs
- `--verbose`: Enable detailed logging

#### Example Use Cases
```bash
# Research publication - high-quality SVG
python scripts/scene_graph_visualizer.py --input-dir outputs/scene_graphs/ --format svg

# Quick preview - batch comparison
python scripts/scene_graph_visualizer.py --input-dir outputs/scene_graphs/ --batch

# Presentation slides - individual PDF graphs
python scripts/scene_graph_visualizer.py --input-dir outputs/scene_graphs/ --format pdf
```

## Key Insights

1. **Perfect Data Alignment**: All SceneFun3D videos exist in ARKitScenes
2. **Coordinate Transformation**: `transform.npy` enables precise alignment
3. **Hierarchical Organization**: 4-level scene graphs support multi-scale reasoning
4. **Motion Integration**: Combines geometric constraints with functional parameters
5. **Confidence Scoring**: Validates object-affordance spatial relationships
6. **Visual Analysis**: Clear hierarchical visualizations enable intuitive scene understanding

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
ModuleNotFoundError: No module named 'numpy'
```
**Solution**: Activate the conda environment
```bash
source /home/jiachen/miniconda3/etc/profile.d/conda.sh
conda activate scenefun3d
```

#### 2. Missing Data Files
```bash
ERROR: Missing required files: [...]
```
**Solution**: Verify data files are copied correctly
```bash
ls -la data_examples/arkitscenes/video_42445781/
ls -la data_examples/scenefun3d/visit_422203/
```

#### 3. Pipeline Execution Fails
```bash
ERROR: Error loading data: [...]
```
**Solution**: Check file permissions and paths
```bash
# Make scripts executable
chmod +x run_pipeline.py validate_outputs.py

# Verify you're in the correct directory
pwd
# Should show: /home/jiachen/scratch/SceneFun3D/alignment
```

#### 4. Open3D Issues
```bash
ImportError: cannot import name 'geometry' from 'open3d'
```
**Solution**: Check Open3D installation
```bash
conda activate scenefun3d
python -c "import open3d as o3d; print(o3d.__version__)"
```

### Performance Notes

- **Point Cloud Loading**: ~2-5 seconds for 2.2M points
- **Spatial Analysis**: ~10-30 seconds for 6 tasks
- **Scene Graph Generation**: ~5-15 seconds per task
- **Total Pipeline**: ~1-3 minutes for complete analysis

### Expected Results Validation

#### Successful Pipeline Output:
```
[INFO] All required input files found
[INFO] Loaded 3 ARKitScenes objects: ['bathtub', 'sink', 'toilet']
[INFO] Loaded 6 SceneFun3D tasks
[INFO] Found 6 spatial relationships
[INFO] Successfully built 6 scene graphs
[INFO] PIPELINE COMPLETED SUCCESSFULLY
```

#### Key Metrics:
- **Objects detected**: 3 (bathtub, sink, toilet)
- **Tasks processed**: 6
- **Spatial relationships**: 6
- **Scene graphs generated**: 6
- **High confidence relationships**: 4-6

This framework enables robust 3D scene understanding that combines geometric precision with functional semantics.

---
**Created**: September 23, 2025
**Data**: Bathroom scene 422203/42445781
**Status**: Production-ready pipeline with validation
**Version**: 1.0