# Models Directory

Place pre-trained object detection model weights here.

## Supported Formats
- `yolov8n.pt`  — YOLOv8 nano (default, ~6MB)
- `yolov5s.pt`  — YOLOv5 small
- Custom ONNX models

## Download YOLOv8 Nano
```bash
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

> **Note:** Model weights are excluded from git (see .gitignore).
> Teammates should download models following the steps above.
