import os
import sys
import unittest
import numpy as np
import cv2
import pandas as pd
from datetime import datetime
import msvcrt

import warnings
warnings.filterwarnings("ignore")

import main

class TestALPRSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load models once for test suite
        cls.plate_model = main.YOLO(main.MODEL_PATH)
        cls.ocr_reader = main.easyocr.Reader(["en"], verbose=False)

    def setUp(self):
        self.test_log = "test_run_log.csv"
        self.test_backup_log = "test_backup_log.csv"
        main.LOG_FILE = self.test_log
        main.BACKUP_LOG_FILE = self.test_backup_log
        for f in (self.test_log, self.test_backup_log):
            if os.path.exists(f):
                os.remove(f)

    def tearDown(self):
        for f in (self.test_log, self.test_backup_log):
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

    def test_01_text_cleaning(self):
        """Test plate text sanitization with various noisy characters."""
        self.assertEqual(main.clean_plate_text("MH-12-DE-1433"), "MH12DE1433")
        self.assertEqual(main.clean_plate_text("  dl 01 ab 1234  "), "DL01AB1234")
        self.assertEqual(main.clean_plate_text("KA.05!MJ?9999"), "KA05MJ9999")
        self.assertEqual(main.clean_plate_text(""), "")
        self.assertEqual(main.clean_plate_text(None), "")

    def test_02_log_detection_normal(self):
        """Test logging to CSV and verifying schema and types."""
        main.log_detection("MH12DE1433", 0.92, log_file=self.test_log)
        self.assertTrue(os.path.exists(self.test_log))
        
        df = pd.read_csv(self.test_log)
        self.assertEqual(len(df), 1)
        self.assertListEqual(list(df.columns), ["Timestamp", "License_Plate", "Confidence"])
        self.assertEqual(df.iloc[0]["License_Plate"], "MH12DE1433")
        self.assertAlmostEqual(df.iloc[0]["Confidence"], 0.92)

    def test_03_log_file_locked_handling(self):
        """Simulate locked file (e.g. open in Excel on Windows) and verify backup logging."""
        with open(self.test_log, "w") as f:
            f.write("Timestamp,License_Plate,Confidence\n")
        
        # Lock the test log exclusively
        lock_fd = open(self.test_log, "r+")
        size = os.path.getsize(self.test_log)
        msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, size + 1000)
        try:
            # Should not raise PermissionError; should fall back to backup log
            main.log_detection("KA01AB1234", 0.88, log_file=self.test_log)
            self.assertTrue(os.path.exists(self.test_backup_log), "Backup log should be created")
            df_backup = pd.read_csv(self.test_backup_log)
            self.assertEqual(len(df_backup), 1)
            self.assertEqual(df_backup.iloc[0]["License_Plate"], "KA01AB1234")
        finally:
            try:
                msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, size + 1000)
            except Exception:
                pass
            lock_fd.close()

    def test_04_preprocess_plate_empty_and_normal(self):
        """Test preprocessing with empty array and normal crop."""
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        t, g = main.preprocess_plate(empty)
        self.assertIsNone(t)
        self.assertIsNone(g)

        # Normal plate crop
        plate_crop = cv2.imread("sample_plate.jpg")
        thresh, gray = main.preprocess_plate(plate_crop)
        self.assertIsNotNone(thresh)
        self.assertIsNotNone(gray)
        self.assertEqual(len(thresh.shape), 2)  # Single channel
        self.assertEqual(len(gray.shape), 2)    # Single channel
        self.assertGreater(thresh.shape[0], 0)
        self.assertGreater(thresh.shape[1], 0)

    def test_05_process_frame_with_plate(self):
        """Test full pipeline end-to-end on test_frame.jpg."""
        frame = cv2.imread("test_frame.jpg")
        recent = {}
        processed = main.process_frame(frame.copy(), self.plate_model, self.ocr_reader, recent)
        
        self.assertIsNotNone(processed)
        self.assertEqual(processed.shape, frame.shape)
        # Verify plate was detected and added to recent detections
        self.assertIn("MH12DE1433", recent)
        # Verify detection was logged
        self.assertTrue(os.path.exists(self.test_log))
        df = pd.read_csv(self.test_log)
        self.assertEqual(df.iloc[0]["License_Plate"], "MH12DE1433")

    def test_06_process_frame_without_plate(self):
        """Test frame with no license plate (e.g. solid background)."""
        blank_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        recent = {}
        processed = main.process_frame(blank_frame.copy(), self.plate_model, self.ocr_reader, recent)
        self.assertIsNotNone(processed)
        self.assertEqual(len(recent), 0)
        # Nothing should be logged
        self.assertFalse(os.path.exists(self.test_log))

    def test_07_cooldown_duplicate_prevention(self):
        """Test that same plate within cooldown interval is not logged twice."""
        frame = cv2.imread("test_frame.jpg")
        recent = {}
        # First call
        main.process_frame(frame.copy(), self.plate_model, self.ocr_reader, recent)
        # Immediate second call
        main.process_frame(frame.copy(), self.plate_model, self.ocr_reader, recent)
        
        df = pd.read_csv(self.test_log)
        # Should only have logged once
        self.assertEqual(len(df), 1)

    def test_08_assemble_plate_text_starting_number(self):
        """Test that multi-segment and starting numbers (e.g., RJ14 + CV0002) are fully assembled."""
        # Simulated EasyOCR output where RJ14 and CV0002 are detected as separate bounding boxes
        simulated_ocr = [
            ([[18, 26], [78, 26], [78, 74], [18, 74]], "RJ", 0.45),
            ([[94, 26], [160, 26], [160, 76], [94, 76]], "14", 0.75),
            ([[180, 26], [380, 26], [380, 76], [180, 76]], "CV 0002", 0.88),
        ]
        plate, conf = main.assemble_plate_text(simulated_ocr)
        self.assertEqual(plate, "RJ14CV0002")
        self.assertGreaterEqual(conf, 0.40)

        # Test two-line plate (Line 1: RJ 14, Line 2: CV 0002)
        simulated_two_line = [
            ([[20, 20], [180, 20], [180, 50], [20, 50]], "RJ 14", 0.80),
            ([[20, 60], [220, 60], [220, 95], [20, 95]], "CV 0002", 0.85),
        ]
        plate_2line, _ = main.assemble_plate_text(simulated_two_line)
        self.assertEqual(plate_2line, "RJ14CV0002")

if __name__ == "__main__":
    unittest.main()
