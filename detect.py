import os
import sys
import time
import numpy as np
import cv2

from tritonclient.http import InferenceServerClient, InferInput
from tritonclient.utils import InferenceServerException
from coco_labels_91 import COCO_LABELS  # labels file must be next to this script

# Configuration
TRITON_URL  = "localhost:8000"      # Triton server address (no "http://")
MODEL_NAME  = "ssd_mobilenet_v1"
IMG_PATH    = "test.jpg"            # replace with your own image
CONF_THRESH = 0.30                  # detection threshold


def main():
    # Quick check for input image
    if not os.path.isfile(IMG_PATH):
        print(f"[!] Image not found: {IMG_PATH}")
        sys.exit(1)

    # Connect to Triton
    try:
        client = InferenceServerClient(url=TRITON_URL, verbose=False)
    except InferenceServerException as e:
        print(f"[!] Triton client error: {e}")
        sys.exit(1)

    # Load image
    img_bgr = cv2.imread(IMG_PATH)
    if img_bgr is None:
        print(f"[!] Failed to read image: {IMG_PATH}")
        sys.exit(1)
    h0, w0 = img_bgr.shape[:2]

    # Preprocess to match model input
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (300, 300))
    input_tensor = img_resized.astype(np.uint8)[None, ...]

    inp = InferInput("image_tensor", input_tensor.shape, "UINT8")
    inp.set_data_from_numpy(input_tensor)

    # Run inference
    print("[*] Sending request to Triton...")
    t0 = time.time()
    try:
        result = client.infer(model_name=MODEL_NAME, inputs=[inp])
    except InferenceServerException as e:
        print(f"[!] Inference error: {e}")
        sys.exit(1)
    dt_ms = (time.time() - t0) * 1000.0
    print(f"[+] Inference finished in {dt_ms:.1f} ms")

    # Parse outputs
    boxes   = result.as_numpy("detection_boxes")
    scores  = result.as_numpy("detection_scores")
    classes = result.as_numpy("detection_classes")
    num     = int(result.as_numpy("num_detections")[0])

    out = img_bgr.copy()
    for i in range(num):
        s = float(scores[0, i])
        if s < CONF_THRESH:
            continue
        y1, x1, y2, x2 = boxes[0, i]
        x1i, y1i = int(x1 * w0), int(y1 * h0)
        x2i, y2i = int(x2 * w0), int(y2 * h0)

        cls_id = int(classes[0, i])
        label = COCO_LABELS[cls_id] if 0 <= cls_id < len(COCO_LABELS) else f"id:{cls_id}"

        cv2.rectangle(out, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
        cv2.putText(out, f"{label} {s:.2f}", (x1i, max(0, y1i - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Show + save
    cv2.imshow("Detections", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("out.jpg", out)
    print("[+] Saved result as out.jpg")


if __name__ == "__main__":
    main()
