# Real-time Object Detection and Tracking with YOLOv8 and OpenCV

## Overview

This project implements a real-time object detection and tracking system using YOLOv8, OpenCV, and MSS for screen capture. The system captures the screen, detects objects within a defined region of interest (ROI), and tracks selected objects.

## Features

- **Real-time screen capture**: Continuously captures the screen and processes the images in real time.
- **Object detection**: Uses YOLOv8 for detecting objects within a user-defined ROI.
- **ROI selection**: Allows the user to draw, resize, and drag a rectangular ROI using mouse interactions.
- **Object tracking**: Tracks selected objects within the ROI using OpenCV's CSRT tracker.

## Dependencies

- Python 3.x
- OpenCV
- NumPy
- MSS
- PyTorch
- Ultralytics YOLOv8

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/yolo-object-detection-tracking.git
    cd yolo-object-detection-tracking
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the YOLOv8 model**:
    Download the `yolov8n.pt` model from the [Ultralytics YOLOv8 repository](https://github.com/ultralytics/yolov8) and place it in the project directory.

## Usage

1. **Run the script**:
    ```bash
    python main.py
    ```

2. **Interact with the window**:
    - **Draw ROI**: Click and drag to draw a rectangular ROI.
    - **Resize ROI**: Click near the corners of the ROI and drag to resize.
    - **Drag ROI**: Click inside the ROI and drag to move.
    - **Select Object for Tracking**: Right-click on the detected object to start tracking. Right-click again to stop tracking.
    - **Quit**: Press `q` to quit the application.

## Code Structure

### YOLOModel Class

Handles loading the YOLOv8 model and running object detection on given images.

### ScreenCaptureView Class

Handles screen capture, mouse events for drawing, resizing, dragging the ROI, and displays the results.

### Controller Class

Manages the main loop, processes frames, and integrates the model and view. It handles the detection results and sets up object tracking.

### Utility Functions

- `clamp(value, min_value, max_value)`: Ensures the value is within the specified range.

## Contributing

Feel free to open issues or submit pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/yolov8)
- [OpenCV](https://opencv.org/)
- [MSS](https://github.com/BoboTiG/python-mss)

## Contact

For any questions, please reach out to akashamalan53@gmail.com.

Enjoy detecting and tracking objects in real time!
