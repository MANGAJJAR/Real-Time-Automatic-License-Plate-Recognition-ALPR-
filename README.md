# Automatic License Plate Recognition (ALPR) System

A robust real-time Automatic License Plate Recognition pipeline powered by **YOLOv8**, **EasyOCR**, **OpenCV**, and **Pandas**.

---

## Architecture & Pipeline

```
Video Stream / Camera (cv2.VideoCapture)
                     │
                     ▼
       YOLOv8 Plate Detection (license_plate_detector.pt)
                     │
                     ▼ [Crop Bounding Box]
       Advanced Image Preprocessing:
       - Smart upscaling for small plates
       - Grayscale conversion
       - Bilateral filtering (noise reduction & edge preservation)
       - Otsu thresholding (maximum text-to-background contrast)
       - Replicate padding (avoids clipping edge characters)
                     │
                     ▼
       EasyOCR Character Recognition & Text Sanitization
                     │
                     ├──► Green Bounding Box & Label Overlay (cv2.imshow)
                     └──► Resilient CSV Logging with Excel Lock Protection (log.csv)
```

---

## Edge Case Safeguards & Features

1. **Windows Excel Lock Resilience:**
   - If `log.csv` is opened in Microsoft Excel or another application that places exclusive locks on Windows files, detections automatically write to `log_backup.csv` instead of crashing the video stream.
2. **Windows DirectShow Acceleration:**
   - Automatically uses `cv2.CAP_DSHOW` on Windows for webcam sources to avoid camera initialization freezes and MSMF delays.
3. **Window Close ('X') Detection:**
   - Detects when the user clicks the window close button (`cv2.WND_PROP_VISIBLE < 1`) to exit cleanly and prevent headless background process leaks.
4. **Enhanced Plate Preprocessing & Full Text Assembly:**
   - Multi-segment reading-order assembly combines separated plate blocks (e.g. `RJ14` and `CV0002` -> `RJ14CV0002`).
   - Margin-padded cropping ensures YOLO bounding boxes do not clip outer characters.
   - Bilateral denoising, CLAHE adaptive contrast, and border padding maximize EasyOCR accuracy.
5. **Duplicate Cooldown:**
   - Prevents duplicate log spamming if the same vehicle remains stationary in the camera view.
6. **Headless & Static Image Support:**
   - Fully supports static images, videos, or headless servers with `--headless` and `--save`.

---

## Usage Guide

### 1. Live Webcam Feed
```bash
python main.py
```
*(Press `q` or `ESC` in the display window to quit)*

### 2. Static Image Recognition
```bash
python main.py test_frame.jpg
```
*(Displays the detected plate and waits until a key is pressed to close)*

### 3. Save Annotated Output & Headless Mode
```bash
python main.py test_frame.jpg --save output_detected.jpg --headless
```

### 4. Run Automated Unit Tests
```bash
python test_alpr.py
```

---

## CSV Log Structure (`log.csv`)

| Timestamp | License_Plate | Confidence |
| :--- | :--- | :--- |
| 2026-10-23 09:43:04 | MH12DE1433 | 0.59 |
