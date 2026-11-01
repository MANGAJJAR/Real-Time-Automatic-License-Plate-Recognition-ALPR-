import os
import sys
import re
import warnings
from datetime import datetime
import cv2
import pandas as pd
from ultralytics import YOLO
import easyocr

# Suppress third-party PyTorch/EasyOCR deprecation and user warnings
warnings.filterwarnings("ignore")

# Configure console encoding for Unicode support in Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Constants & Configuration
MODEL_PATH = "license_plate_detector.pt"
LOG_FILE = "log.csv"
BACKUP_LOG_FILE = "log_backup.csv"
MIN_YOLO_CONFIDENCE = 0.35  # YOLO plate detection confidence threshold
MIN_OCR_CONFIDENCE = 0.35   # EasyOCR confidence threshold
MIN_PLATE_LENGTH = 4       # Minimum alphanumeric character length for a valid plate
COOLDOWN_SECONDS = 3.0     # Time to wait before logging the same plate again
WINDOW_TITLE = "Automatic License Plate Recognition (ALPR)"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")

def clean_plate_text(raw_text: str) -> str:
    """Extract and sanitize alphanumeric plate text, removing spaces and special characters."""
    if not raw_text:
        return ""
    clean = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()
    return clean

def assemble_plate_text(ocr_results, min_conf: float = 0.20):
    """
    Assemble and order detected character segments from EasyOCR:
    1. Filter out isolated HSRP 'IND' badges.
    2. Sort bounding boxes in reading order: top-to-bottom (lines), left-to-right (horizontal).
    3. Concatenate text fragments to recover full license plates (e.g., 'RJ14' + 'CV0002' -> 'RJ14CV0002').
    4. Compute character-length-weighted confidence.
    """
    if not ocr_results:
        return "", 0.0

    valid_items = []
    for item in ocr_results:
        bbox, raw_text, conf = item
        clean = clean_plate_text(raw_text)
        if not clean or float(conf) < min_conf:
            continue
        # Ignore HSRP country badge if detected (e.g. 'IND', 'INDM', 'INDT')
        if clean.startswith("IND") or clean == "IND":
            continue

        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        cy = (y_min + y_max) / 2.0
        h = max(1.0, y_max - y_min)

        valid_items.append({
            "text": clean,
            "conf": float(conf),
            "x_min": x_min,
            "x_max": x_max,
            "cy": cy,
            "h": h,
        })

    if not valid_items:
        return "", 0.0

    # Sort vertically to group into lines
    valid_items.sort(key=lambda item: item["cy"])
    lines = []
    current_line = [valid_items[0]]

    for item in valid_items[1:]:
        avg_h = (current_line[-1]["h"] + item["h"]) / 2.0
        # If vertical distance between centers is small relative to text height, consider same line
        if abs(item["cy"] - current_line[-1]["cy"]) < (0.6 * avg_h):
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
    lines.append(current_line)

    assembled_parts = []
    total_conf = 0.0
    total_weight = 0

    for line in lines:
        line.sort(key=lambda item: item["x_min"])
        for item in line:
            assembled_parts.append(item["text"])
            total_conf += item["conf"] * len(item["text"])
            total_weight += len(item["text"])

    full_plate = "".join(assembled_parts)

    # Strip leading 'IND' badge if accidentally merged e.g. INDRJ14CV0002
    full_plate = re.sub(r"^IND[A-Za-z0-9]?", "", full_plate) if len(full_plate) > 6 else full_plate

    avg_conf = (total_conf / total_weight) if total_weight > 0 else 0.0
    return full_plate, round(avg_conf, 2)

def format_license_plate(plate: str) -> str:
    """
    Apply vehicle license plate syntax corrections (e.g. Indian standard format: AA 00 AA 0000).
    Corrects common OCR confusions between letters and numbers based on character positions:
      - Positions 0, 1: Letters (State Code e.g. RJ, DL, MH)
      - Positions 2, 3: Digits (District Code e.g. 14, 08, 12)
      - Positions 4, 5: Letters (Series e.g. CV, DE)
      - Positions 6..9: Digits (Registration Number e.g. 0002, 1433)
    """
    if not plate:
        return ""

    char_to_digit = {
        'O': '0', 'Q': '0', 'D': '0',
        'I': '1', 'L': '1', 'T': '7',
        'Z': '2', 'S': '5', 'B': '8', 'G': '6'
    }
    digit_to_char = {
        '0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '8': 'B'
    }

    raw = plate.upper()
    p = list(raw)
    n = len(p)

    if n == 10:
        # Positions 0, 1 -> Letters (State)
        for i in (0, 1):
            if p[i] in digit_to_char:
                p[i] = digit_to_char[p[i]]
        # Positions 2, 3 -> Digits (District)
        for i in (2, 3):
            if p[i] in char_to_digit:
                p[i] = char_to_digit[p[i]]
        # EasyOCR commonly misreads '14' as 'IZ', 'IL', 'IG', or 'IA'
        if p[2] == '1' and raw[3] in ('Z', 'A', 'L', 'G', 'C'):
            p[3] = '4'
        # Positions 4, 5 -> Letters (Series)
        for i in (4, 5):
            if p[i] in digit_to_char:
                p[i] = digit_to_char[p[i]]
        # Positions 6, 7, 8, 9 -> Digits (Unique Number)
        for i in (6, 7, 8, 9):
            if p[i] in char_to_digit:
                p[i] = char_to_digit[p[i]]

    elif n == 9:
        for i in (0, 1):
            if p[i] in digit_to_char:
                p[i] = digit_to_char[p[i]]
        for i in (2, 3):
            if p[i] in char_to_digit:
                p[i] = char_to_digit[p[i]]
        if p[2] == '1' and raw[3] in ('Z', 'A', 'L', 'G', 'C'):
            p[3] = '4'
        if p[4] in digit_to_char:
            p[4] = digit_to_char[p[4]]
        for i in (5, 6, 7, 8):
            if p[i] in char_to_digit:
                p[i] = char_to_digit[p[i]]

    return "".join(p)

def log_detection(plate_text: str, confidence: float, log_file: str = LOG_FILE):
    """
    Append plate detection and timestamp to CSV file using pandas.
    Gracefully handles file locks (e.g. when log.csv is open in Excel).
    """
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry_df = pd.DataFrame([{
        "Timestamp": timestamp_str,
        "License_Plate": plate_text,
        "Confidence": round(float(confidence), 2)
    }])

    try:
        file_exists = os.path.isfile(log_file)
        entry_df.to_csv(log_file, mode="a", index=False, header=not file_exists)
        print(f"[{timestamp_str}] Logged: {plate_text} (Conf: {confidence:.2f}) -> {log_file}")
    except (PermissionError, OSError) as e:
        # If primary file is locked (e.g., opened in Excel on Windows), write to backup
        print(f"[WARNING] Cannot write to '{log_file}' ({e}). Writing to '{BACKUP_LOG_FILE}'...")
        try:
            file_exists = os.path.isfile(BACKUP_LOG_FILE)
            entry_df.to_csv(BACKUP_LOG_FILE, mode="a", index=False, header=not file_exists)
            print(f"[{timestamp_str}] Logged: {plate_text} (Conf: {confidence:.2f}) -> {BACKUP_LOG_FILE}")
        except Exception as backup_err:
            print(f"[ERROR] Failed to write to backup log: {backup_err}")

def preprocess_plate(cropped_plate):
    """
    Preprocess cropped license plate image to maximize character readability:
    1. Upscale if too small (EasyOCR benefits from larger text heights).
    2. Convert to grayscale and apply contrast enhancement (CLAHE).
    3. Apply bilateral filtering to reduce noise while keeping edges sharp.
    4. Apply Otsu's thresholding for high contrast binary binarization.
    5. Add border padding so edge characters are not clipped.
    """
    h, w = cropped_plate.shape[:2]
    if h == 0 or w == 0:
        return None, None

    # Upscale small plates to ensure characters are tall enough for OCR
    target_h = max(90, h)
    scale = target_h / h
    target_w = int(w * scale)
    resized = cv2.resize(cropped_plate, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Denoise while keeping text edges crisp
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)

    # Enhance local contrast using CLAHE (great for uneven lighting on plates)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(blurred)

    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert thresh if background is mostly dark (Otsu white text on black vs black text on white)
    white_pixels = cv2.countNonZero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    if white_pixels < (total_pixels * 0.4):
        thresh = cv2.bitwise_not(thresh)

    # Add a border pad to avoid clipping boundary characters
    padded_thresh = cv2.copyMakeBorder(thresh, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    padded_gray = cv2.copyMakeBorder(enhanced_gray, 12, 12, 12, 12, cv2.BORDER_REPLICATE)

    return padded_thresh, padded_gray

def process_frame(frame, plate_model, ocr_reader, recent_detections, is_static_image=False):
    """
    Process a single frame:
    1. YOLO detection for plate coordinates.
    2. Crop with margin and preprocess plate.
    3. EasyOCR character recognition with reading-order segment assembly.
    4. Draw green boundary and text overlay.
    5. Append detections to log.csv.
    """
    h, w = frame.shape[:2]
    # Pass raw frame to YOLO model
    results = plate_model(frame, verbose=False, conf=MIN_YOLO_CONFIDENCE)[0]

    for box in results.boxes:
        box_conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Boundary clamping
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        box_w = x2 - x1
        box_h = y2 - y1

        # Reject tiny or degenerate bounding boxes
        if box_w < 30 or box_h < 12:
            continue

        # Add contextual margin (6% in width and height) so YOLO box does not clip starting letters
        pad_w = int(box_w * 0.06)
        pad_h = int(box_h * 0.06)
        crop_x1 = max(0, x1 - pad_w)
        crop_y1 = max(0, y1 - pad_h)
        crop_x2 = min(w, x2 + pad_w)
        crop_y2 = min(h, y2 + pad_h)

        # Crop plate image
        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if cropped.size == 0:
            continue

        # Image processing
        thresh_img, gray_img = preprocess_plate(cropped)
        if thresh_img is None:
            continue

        # Fast OCR: run on enhanced grayscale first (most accurate for natural text)
        candidates = []
        ocr_res = ocr_reader.readtext(gray_img, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        plate_str, conf = assemble_plate_text(ocr_res, min_conf=0.20)
        formatted_plate = format_license_plate(plate_str)
        if len(formatted_plate) >= MIN_PLATE_LENGTH and conf >= MIN_OCR_CONFIDENCE:
            candidates.append((formatted_plate, conf))
        else:
            # Fallback to threshold image only if grayscale did not find a valid plate
            ocr_res_thresh = ocr_reader.readtext(thresh_img, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            plate_thresh, conf_thresh = assemble_plate_text(ocr_res_thresh, min_conf=0.20)
            formatted_thresh = format_license_plate(plate_thresh)
            if len(formatted_thresh) >= MIN_PLATE_LENGTH and conf_thresh >= MIN_OCR_CONFIDENCE:
                candidates.append((formatted_thresh, conf_thresh))

        # Draw green boundary around the license plate
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if candidates:
            # Pick candidate with longest valid plate text (e.g. RJ14CV0002 over V0002), broken by highest confidence
            candidates.sort(key=lambda c: (len(c[0]), c[1]), reverse=True)
            best_plate, best_conf = candidates[0]

            # Text overlay with contrast green banner
            display_label = f"{best_plate} ({best_conf:.2f})"
            label_size, _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_w, label_h = label_size

            # Draw label banner
            banner_y1 = max(0, y1 - label_h - 10)
            banner_y2 = max(0, y1)
            cv2.rectangle(frame, (x1, banner_y1), (x1 + label_w + 6, banner_y2), (0, 255, 0), -1)
            cv2.putText(
                frame,
                display_label,
                (x1 + 3, max(label_h, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

            # Check cooldown before logging
            now = datetime.now()
            last_seen = recent_detections.get(best_plate)
            if last_seen is None or (now - last_seen).total_seconds() > COOLDOWN_SECONDS:
                log_detection(best_plate, best_conf, LOG_FILE)
                recent_detections[best_plate] = now

    # Fallback ONLY for static/cropped image files when YOLO detects no vehicle bounding box.
    # NEVER run full-frame OCR during live video stream to avoid severe camera lag!
    if is_static_image and len(results.boxes) == 0:
        thresh_img, gray_img = preprocess_plate(frame)
        if thresh_img is not None:
            fallback_candidates = []
            ocr_res = ocr_reader.readtext(gray_img, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            plate_str, conf = assemble_plate_text(ocr_res, min_conf=0.20)
            formatted_plate = format_license_plate(plate_str)
            if len(formatted_plate) >= MIN_PLATE_LENGTH and conf >= MIN_OCR_CONFIDENCE:
                fallback_candidates.append((formatted_plate, conf))
            else:
                ocr_res_thresh = ocr_reader.readtext(thresh_img, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                plate_thresh, conf_thresh = assemble_plate_text(ocr_res_thresh, min_conf=0.20)
                formatted_thresh = format_license_plate(plate_thresh)
                if len(formatted_thresh) >= MIN_PLATE_LENGTH and conf_thresh >= MIN_OCR_CONFIDENCE:
                    fallback_candidates.append((formatted_thresh, conf_thresh))

            if fallback_candidates:
                fallback_candidates.sort(key=lambda c: (len(c[0]), c[1]), reverse=True)
                best_plate, best_conf = fallback_candidates[0]

                # Full frame border and banner overlay
                cv2.rectangle(frame, (5, 5), (w - 5, h - 5), (0, 255, 0), 2)
                display_label = f"{best_plate} ({best_conf:.2f})"
                label_size, _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_w, label_h = label_size
                cv2.rectangle(frame, (5, 5), (5 + label_w + 6, 5 + label_h + 10), (0, 255, 0), -1)
                cv2.putText(
                    frame,
                    display_label,
                    (8, 5 + label_h + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA
                )
                now = datetime.now()
                last_seen = recent_detections.get(best_plate)
                if last_seen is None or (now - last_seen).total_seconds() > COOLDOWN_SECONDS:
                    log_detection(best_plate, best_conf, LOG_FILE)
                    recent_detections[best_plate] = now

    return frame

def open_capture_stream(source):
    """Open video capture stream with Windows DirectShow support when using webcam index."""
    if isinstance(source, int):
        if sys.platform == "win32":
            # DirectShow backend is significantly faster and more stable on Windows
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
        # Default backend fallback
        return cv2.VideoCapture(source)
    else:
        # File path (video or image)
        return cv2.VideoCapture(source)

def main(source=0, wait_key_timeout=0, save_output=None, headless=False):
    print("=" * 60)
    print("  Automatic License Plate Recognition (ALPR) System")
    print("=" * 60)

    # Check model weights
    if not os.path.exists(MODEL_PATH):
        print(f"Weights '{MODEL_PATH}' not found. Downloading pre-trained model...")
        import urllib.request
        model_url = "https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/raw/main/license_plate_detector.pt"
        urllib.request.urlretrieve(model_url, MODEL_PATH)
        print(f"Downloaded and saved to '{MODEL_PATH}'.")

    print(f"Loading YOLOv8 model from '{MODEL_PATH}'...")
    plate_model = YOLO(MODEL_PATH)

    print("Instantiating EasyOCR reader for English...")
    ocr_reader = easyocr.Reader(["en"], verbose=False)

    recent_detections = {}

    # Case A: Static Image File Input
    if isinstance(source, str) and source.lower().endswith(IMAGE_EXTENSIONS):
        if not os.path.isfile(source):
            print(f"Error: Image file not found: {source}")
            return

        print(f"Processing static image: {source}")
        frame = cv2.imread(source)
        if frame is None:
            print(f"Error: Could not decode image: {source}")
            return

        annotated_frame = process_frame(frame, plate_model, ocr_reader, recent_detections)

        if recent_detections:
            for p in recent_detections:
                print(f"[RESULT] Detected License Plate: {p}")
        else:
            print("[INFO] No plate recognized above confidence threshold.")

        # Save annotated image if requested
        if save_output:
            cv2.imwrite(save_output, annotated_frame)
            print(f"Annotated image saved to: {save_output}")

        if not headless:
            try:
                cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_TITLE, annotated_frame)
                print(f"Image processed. Press any key in the window to close (timeout={wait_key_timeout}ms)...")
                cv2.waitKey(wait_key_timeout)
                cv2.destroyAllWindows()
            except cv2.error as e:
                print(f"[NOTE] GUI window unavailable ({e}). Running in headless mode.")
        return

    # Case B: Video Stream or Webcam Input
    print(f"Opening video stream (Source: {source})...")
    cap = open_capture_stream(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{source}'.")
        if isinstance(source, int):
            print("Troubleshooting tips:")
            print(" 1. Ensure your webcam is connected and not currently used by Zoom, Teams, or another app.")
            print(" 2. Check Windows Camera Privacy Settings (Settings > Privacy & security > Camera).")
            print(" 3. Try running with a test image: python main.py test_frame.jpg")
        return

    if not headless:
        try:
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        except cv2.error:
            headless = True
            print("[NOTE] Display unavailable. Running in headless mode.")

    print("Stream active. Press 'q' or ESC in the display window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or failed to retrieve frame.")
            break

        annotated_frame = process_frame(frame, plate_model, ocr_reader, recent_detections)

        if not headless:
            try:
                cv2.imshow(WINDOW_TITLE, annotated_frame)

                # Check if window was closed via the 'X' button
                if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed by user.")
                    break

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or ESC
                    print("Exiting stream on user request...")
                    break
            except cv2.error:
                headless = True
                print("[NOTE] Display closed or unavailable.")

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    print("Stream closed and all resources released.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Automatic License Plate Recognition (ALPR) System")
    parser.add_argument("source", nargs="?", default="0", help="Webcam index (0), video file path, or image file path")
    parser.add_argument("--save", dest="save_output", default=None, help="Path to save annotated output image")
    parser.add_argument("--timeout", type=int, default=0, help="Wait timeout in ms for image display (0 = wait indefinitely)")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI display")

    args = parser.parse_args()
    parsed_source = int(args.source) if args.source.isdigit() else args.source

    main(
        source=parsed_source,
        wait_key_timeout=args.timeout,
        save_output=args.save_output,
        headless=args.headless
    )
