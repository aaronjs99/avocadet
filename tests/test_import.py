import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    from avocadet_lib import UnifiedDetector

    print("Successfully imported UnifiedDetector")
except ImportError as e:
    print(f"Failed to import UnifiedDetector: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
