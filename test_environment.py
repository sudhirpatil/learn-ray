#!/usr/bin/env python3
"""
Test script to verify the virtual environment and Ray installation
"""

import sys
import os

def main():
    print("=== Environment Test ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Virtual environment: {os.environ.get('VIRTUAL_ENV', 'Not activated')}")
    
    try:
        import ray
        print(f"Ray version: {ray.__version__}")
        print("✅ Ray[data] is successfully installed!")
        
        # Test basic Ray functionality
        ray.init(ignore_reinit_error=True)
        print("✅ Ray initialization successful!")
        
        # Test Ray Data
        import ray.data as ray_data
        print("✅ Ray Data module imported successfully!")
        
        ray.shutdown()
        print("✅ Ray shutdown successful!")
        
    except ImportError as e:
        print(f"❌ Error importing Ray: {e}")
    except Exception as e:
        print(f"❌ Error with Ray: {e}")

if __name__ == "__main__":
    main()

