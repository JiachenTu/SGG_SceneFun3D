# Scene Graph Generation Pipeline V2.0 🚀

Enhanced pipeline for generating hierarchical scene graphs with floating affordance support.

## ✨ Key Features

- **Coordinate System Inversion**: Transform ARKit objects to SceneFun3D space
- **Official SceneFun3D Integration**: Uses official DataParser toolkit
- **No Fallback Logic**: Honest uncertainty handling with floating affordances
- **Configurable**: Works with any data root path
- **Test-First Approach**: Comprehensive testing before integration

## 🏗️ Architecture

```
📁 V2.0 Pipeline Components
├── 🧪 Testing & Validation
│   ├── test_official_parser.py          # Test SceneFun3D DataParser
│   └── test_coordinate_transform.py     # Test transformations
├── 🔧 Core Utilities
│   ├── unified_data_loader.py           # Configurable data loading
│   ├── enhanced_coordinate_transform.py # ARKit → SceneFun3D transformation
│   └── spatial_scorer.py                # Simple confidence scoring
├── 🏗️ Scene Graph Generation
│   └── scene_graph_builder_v2.py        # Enhanced graph builder
├── 🚀 Main Pipeline
│   ├── run_pipeline_v2.py               # Main orchestration
│   └── config_v2.yaml                   # Configuration template
```

## 🚀 Quick Start

### 1. Test Components First

```bash
# Test official SceneFun3D DataParser
python test_official_parser.py --data-root /path/to/scenefun3d/data

# Test coordinate transformations
python test_coordinate_transform.py --data-root /path/to/scenefun3d/data
```

### 2. Configure Pipeline

```bash
# Copy and edit configuration
cp config_v2.yaml my_config.yaml
# Edit paths in my_config.yaml
```

### 3. Run Pipeline

```bash
# Using configuration file
python run_pipeline_v2.py --config my_config.yaml

# Or using command line arguments
python run_pipeline_v2.py \
    --data-root /path/to/scenefun3d/data \
    --arkitscenes-file /path/to/arkitscenes/annotation.json \
    --visit-id 422203 \
    --video-id 42445781 \
    --output-dir outputs_v2
```

## 📊 Expected Output

```
outputs_v2/
├── scene_graphs/                    # Generated scene graphs
│   ├── 422203_task_01_close_door.json
│   ├── 422203_task_02_flush_toilet.json
│   └── ...
├── pipeline_results_v2.json        # Complete results & statistics
└── pipeline_v2.log                 # Execution log
```

## ⚙️ Configuration

Key settings in `config_v2.yaml`:

```yaml
spatial_analysis:
  confidence_threshold: 0.3    # Minimum confidence for relationships
  allow_floating: true         # Enable floating affordances

data:
  root_path: "/path/to/data"   # Any SceneFun3D data root
```

## 🎯 Key Improvements Over V1.0

- ✅ **Works with any data path** (not hardcoded)
- ✅ **No false connections** (confidence threshold enforced)
- ✅ **Floating affordances** (honest uncertainty)
- ✅ **Official toolkit integration** (SceneFun3D DataParser)
- ✅ **Comprehensive testing** (validate before running)
- ✅ **Simple confidence scoring** (overlap + distance)

## 🧪 Testing Strategy

1. **Test SceneFun3D DataParser** → Ensures data loading works
2. **Test coordinate transforms** → Ensures accuracy
3. **Test spatial scoring** → Ensures reasonable confidence
4. **Integration test** → Full pipeline validation

## 📈 Success Metrics

- **No hardcoded paths**: ✅ Configurable data roots
- **No fallback connections**: ✅ Strict confidence thresholds
- **Floating affordances**: ✅ Support for unattached affordances
- **Fast processing**: ✅ Target <30s per scene
- **High accuracy**: ✅ Round-trip transform error <1e-6

## 🔧 Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed
2. **Data not found**: Check paths in configuration
3. **Transform errors**: Run coordinate transform test first
4. **Low confidence**: Adjust `confidence_threshold` in config

### Debug Mode

```bash
# Run with verbose logging
python run_pipeline_v2.py --config my_config.yaml --verbose

# Test individual components
python test_official_parser.py --data-root /path/to/data
python test_coordinate_transform.py --data-root /path/to/data
```

## 🎉 Ready to Use!

The V2.0 pipeline is designed to be:
- **Simple**: Clear, focused components
- **Reliable**: Test-first approach
- **Honest**: No false connections
- **Flexible**: Works with any data root

Just test the components first, then run the pipeline! 🚀