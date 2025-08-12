#!/usr/bin/env python3
"""
Script to view Trackio experiment tracking dashboard.
"""

try:
    import trackio
    TRACKING_AVAILABLE = True
except ImportError:
    print("Trackio not installed. Install with 'pip install trackio' for experiment tracking.")
    TRACKING_AVAILABLE = False

if __name__ == "__main__":
    if TRACKING_AVAILABLE:
        # Show the dashboard
        trackio.show()
    else:
        print("Please install trackio to view the dashboard:")
        print("  pip install trackio")