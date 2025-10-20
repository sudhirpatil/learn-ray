#!/usr/bin/env python3
"""
Test script to verify Ray configuration and memory settings
"""

import os
import ray

def main():
    print("=== Ray Configuration Test ===")
    
    # Check environment variables
    ray_memory_prop = os.environ.get('RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION')
    print(f"RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION: {ray_memory_prop}")
    
    # Initialize Ray with current settings
    try:
        ray.init(ignore_reinit_error=True)
        print("✅ Ray initialized successfully!")
        
        # Get Ray cluster resources
        resources = ray.cluster_resources()
        print(f"Ray cluster resources: {resources}")
        
        # Get object store memory info
        if 'object_store_memory' in resources:
            object_store_memory = resources['object_store_memory']
            print(f"Object store memory: {object_store_memory / (1024**3):.2f} GB")
        
        ray.shutdown()
        print("✅ Ray shutdown successful!")
        
    except Exception as e:
        print(f"❌ Error with Ray: {e}")

if __name__ == "__main__":
    main()
