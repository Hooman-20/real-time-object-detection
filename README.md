# Real-Time Object Detection with NVIDIA Triton

This repo shows how I set up a real-time object detection system using **NVIDIA Triton Inference Server**.  
The model used here is **SSD MobileNet v1** (trained on COCO), and I tested it both on single images and with a live webcam feed.

---


## Features
- Runs object detection in real time (SSD MobileNet v1, COCO dataset)
- Works on both:
  - images (`detect.py`)
  - live webcam (`detect_cam.py`)
- Shows bounding boxes, class labels, and confidence scores
- Uses Triton with TensorRT + CUDA for GPU acceleration


---

## Project Files
- `detect.py` – run detection on an image
- `detect_cam.py` – run detection on webcam (press **q** to quit)
- `coco_labels_91.py` – COCO labels (91 classes)
- `requirements.txt` – dependencies
- `triton_model_repo/` – model folder for Triton

---

## How to Run

1. **Start Triton Server**

   Make sure Docker is running and start Triton like this:

   ```bash
   docker run --rm --gpus all \
     -p 8000:8000 -p 8001:8001 -p 8002:8002 \
     -v $(pwd)/triton_model_repo:/models \
     nvcr.io/nvidia/tritonserver:23.12-py3 tritonserver --model-repository=/models


2. Install Dependencies
pip install -r requirements.txt

3. Run Detection

On an image:
```
python detect.py
```

Update IMG_PATH in the script to your test image.

Live webcam:
```
python detect_cam.py
```

Press q to quit.

---

## Example

Detects objects such as person, car, dog, laptop, bottle, etc.

Outputs bounding boxes with class names and confidence scores.

---

## Example Output

Detection on an Image:
![out](https://github.com/user-attachments/assets/72d16a90-3e43-442d-966a-32ccc5042265)


Live Webcam Detection:


<img width="625" height="472" alt="Screenshot 2025-09-27 164523" src="https://github.com/user-attachments/assets/a406bc26-c6ce-41a6-ab7d-2fdaa13421ec" />



---
## Tech Stack

NVIDIA Triton Inference Server

TensorRT + CUDA (GPU acceleration)

OpenCV + NumPy

Python 3
