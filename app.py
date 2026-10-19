
import gradio as gr
import numpy as np
import torch
import easyocr
import cv2
from ultralytics import YOLO


MODEL_PATH = "best.pt"

model = YOLO(MODEL_PATH)

reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())


def anpr_pipeline(image):
    if image is None:
        return None, "No image provided", "N/A", "Please upload an image."

    results = model(image)
    result = results[0]

    if len(result.boxes) == 0:
        return image, "No license plate detected", "N/A", "No license plate was detected."

    confidence = float(result.boxes.conf[0])

    image_np = np.array(image)
    h, w = image_np.shape[:2]

    x1, y1, x2, y2 = result.boxes.xyxy[0].cpu().numpy().astype(int)

    # Keep coordinates inside the image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    plate_crop = image_np[y1:y2, x1:x2]

    if plate_crop.size == 0:
        return image, "Invalid plate crop", "N/A", "The detected plate crop is empty."

    # Enlarge the plate crop 4x before OCR
    enlarged_crop = cv2.resize(
        plate_crop,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    ocr_results = reader.readtext(enlarged_crop)

    if ocr_results:
        # Select the tallest OCR text box
        best_ocr = max(
            ocr_results,
            key=lambda item: max(point[1] for point in item[0])
            - min(point[1] for point in item[0])
        )

        ocr_text = best_ocr[1]
        ocr_confidence = float(best_ocr[2])

        if ocr_confidence < 0.30:
            ocr_text = "Low confidence: " + ocr_text
    else:
        ocr_text = "No text detected"
        ocr_confidence = None

    # YOLO returns BGR; Gradio expects RGB
    annotated_image = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

    yolo_confidence = round(confidence, 3)

    if ocr_confidence is not None:
        ocr_confidence_text = round(ocr_confidence, 3)
    else:
        ocr_confidence_text = "N/A"

    status = (
        "License Plate Detected\n"
        f"YOLO Confidence: {yolo_confidence}\n"
        f"OCR Text: {ocr_text}\n"
        f"OCR Confidence: {ocr_confidence_text}"
    )

    return annotated_image, ocr_text, str(ocr_confidence_text), status


demo = gr.Interface(
    fn=anpr_pipeline,
    inputs=gr.Image(type="pil", label="Upload Vehicle Image"),
    outputs=[
        gr.Image(label="Detection Result"),
        gr.Textbox(label="OCR Text"),
        gr.Textbox(label="OCR Confidence"),
        gr.Textbox(label="Status")
    ],
    title="Automatic Number Plate Recognition (ANPR)",
    description="Upload a vehicle image to detect a license plate and recognize its text."
)


if __name__ == "__main__":
    demo.launch()
