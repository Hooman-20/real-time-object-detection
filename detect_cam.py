import cv2
import time
import numpy as np
from tritonclient.http import InferenceServerClient, InferInput
from tritonclient.utils import InferenceServerException
from coco_labels_91 import COCO_LABELS

# Configuration
TRITON_URL  = "localhost:8000"
MODEL_NAME  = "ssd_mobilenet_v1"
CONF_THRESH = 0.30


def main():
    # Connect to Triton
    try:
        client = InferenceServerClient(url=TRITON_URL, verbose=False)
    except InferenceServerException as e:
        print(f"[!] Triton client error: {e}")
        return

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Could not open webcam")
        return

    print("[*] Live object detection started (press 'q' to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h0, w0 = frame.shape[:2]

        # Preprocess frame
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (300, 300))
        input_tensor = img_resized.astype(np.uint8)[None, ...]

        inp = InferInput("image_tensor", input_tensor.shape, "UINT8")
        inp.set_data_from_numpy(input_tensor)

        # Inference
        t0 = time.time()
        try:
            result = client.infer(model_name=MODEL_NAME, inputs=[inp])
        except InferenceServerException as e:
            print(f"[!] Inference error: {e}")
            break
        dt_ms = (time.time() - t0) * 1000.0

        # Parse detections
        boxes   = result.as_numpy("detection_boxes")
        scores  = result.as_numpy("detection_scores")
        classes = result.as_numpy("detection_classes")
        num     = int(result.as_numpy("num_detections")[0])

        for i in range(num):
            s = float(scores[0, i])
            if s < CONF_THRESH:
                continue
            y1, x1, y2, x2 = boxes[0, i]
            x1i, y1i = int(x1 * w0), int(y1 * h0)
            x2i, y2i = int(x2 * w0), int(y2 * h0)

            cls_id = int(classes[0, i])
            label = COCO_LABELS[cls_id] if 0 <= cls_id < len(COCO_LABELS) else f"id:{cls_id}"

            cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {s:.2f}", (x1i, max(0, y1i - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Overlay latency
        cv2.putText(frame, f"{dt_ms:.1f} ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Live Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
