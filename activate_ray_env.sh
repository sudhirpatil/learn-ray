#!/bin/bash
# Script to activate the virtual environment with Ray configuration

# Set Ray environment variables
export RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.5
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1

# Activate virtual environment
source .venv/bin/activate

echo "✅ Ray environment activated with 50% memory allocation for object store"
echo "RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION: $RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION"
