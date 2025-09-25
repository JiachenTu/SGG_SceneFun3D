#!/bin/bash
# Setup script for SceneFun3D & ARKitScenes alignment pipeline

echo "SceneFun3D & ARKitScenes Alignment Pipeline Setup"
echo "================================================"

# Check if we're in the right directory
if [ ! -d "data_examples" ]; then
    echo "ERROR: data_examples directory not found."
    echo "Please run this script from the alignment directory."
    exit 1
fi

# Check conda environment
echo "Checking conda environment..."
if command -v conda &> /dev/null; then
    echo "✓ Conda found"
else
    echo "✗ Conda not found. Please install Miniconda/Anaconda."
    exit 1
fi

# Activate scenefun3d environment
echo "Activating scenefun3d environment..."
source /home/jiachen/miniconda3/etc/profile.d/conda.sh
conda activate scenefun3d

# Check Python dependencies
echo "Checking Python dependencies..."
python -c "
import sys
import subprocess

required_packages = ['numpy', 'open3d', 'json']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
        print(f'✓ {package} available')
    except ImportError:
        missing_packages.append(package)
        print(f'✗ {package} missing')

if missing_packages:
    print(f'Missing packages: {missing_packages}')
    sys.exit(1)
else:
    print('All required packages available')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Missing required Python packages."
    echo "Please install them in the scenefun3d environment."
    exit 1
fi

# Test data files
echo "Checking data files..."
data_files=(
    "data_examples/arkitscenes/video_42445781/42445781_3dod_annotation.json"
    "data_examples/scenefun3d/visit_422203/422203_descriptions.json"
    "data_examples/scenefun3d/visit_422203/422203_annotations.json"
    "data_examples/scenefun3d/visit_422203/422203_motions.json"
    "data_examples/scenefun3d/visit_422203/422203_laser_scan.ply"
    "data_examples/scenefun3d/visit_422203/42445781/42445781_transform.npy"
)

all_files_exist=true
for file in "${data_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file"
        all_files_exist=false
    fi
done

if [ "$all_files_exist" = false ]; then
    echo "ERROR: Some required data files are missing."
    exit 1
fi

# Make scripts executable
echo "Making scripts executable..."
chmod +x run_pipeline.py validate_outputs.py

# Test basic functionality
echo "Testing basic functionality..."
python utils/arkitscenes_parser.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ ARKitScenes parser working"
else
    echo "✗ ARKitScenes parser failed"
    exit 1
fi

python utils/scenefun3d_parser.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ SceneFun3D parser working"
else
    echo "✗ SceneFun3D parser failed"
    exit 1
fi

echo ""
echo "Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Run the pipeline: python run_pipeline.py --validate --verbose"
echo "2. Check outputs in: outputs/"
echo "3. Validate results: python validate_outputs.py --verbose"
echo ""
echo "For help, see: README.md"