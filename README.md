# FlowTracker: Sparse Optical Flow

## Overview

This repository provides a Python implementation for tracking and visualizing optical flow in video sequences. The `FlowTracker` application identifies prominent feature points and tracks their continuous movement across consecutive frames, rendering colorful motion trails in real-time. 

## Key Features

* **Lucas-Kanade Tracking:** Computes sparse optical flow using `cv2.calcOpticalFlowPyrLK` to determine feature displacement between frames.
* **Dynamic Feature Initialization:** Detects strong tracking corners at startup and automatically re-initializes anchors using `cv2.goodFeaturesToTrack` if the number of surviving points drops below 5.
* **Real-Time Visualization:** Renders distinct motion trajectories using a randomized 100-color palette, overlaid on a live display window named "optical flow".
* **On-Screen Diagnostics:** Projects a live count of currently tracked points directly onto the video canvas.

## Prerequisites

Ensure the following dependencies are installed in your Python environment:
* `numpy`[cite: 1]
* `opencv-python` (imported as `cv2`)
* Standard libraries: `argparse`, `sys`

## Quick Start

To launch the tracker, execute `main.py` from your terminal, passing the path to your target video file via the `--video` argument.

```bash
python main.py --video path/to/your/video.mp4
```

## Controls
Controls
While the visualization window is active:

* **q or ESC**: Gracefully terminate the application and close all windows.

## Architecture & Configuration
The core tracking pipeline is encapsulated within the FlowTracker class found in main.py. It relies on the following configurations:

* Corner Detection (_CORNER_CFG): Configured to identify a maximum of 100 corners, with a quality level of 0.3, a minimum distance of 7, and a block size of 7.

* Optical Flow (_LK_CFG): The Lucas-Kanade algorithm uses a 15x15 window size, a maximum of 2 pyramid levels, and an epsilon/count-based termination criteria.
