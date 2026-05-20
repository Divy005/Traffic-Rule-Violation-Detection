"""
solution.py — AID 728 Traffic Rule Violation Detection
=======================================================
Pipeline:
  1. YOLOv8s (COCO) + custom bike detector  →  bike boxes + person boxes
  2. Depth-Anything V2 (fp16)               →  depth map for person→bike association
  3. Helmet classifier (YOLO)                →  helmet / no-helmet per rider
  4. license.pt (YOLO)                       →  license plate bounding box
  5. PaddleOCR 3.5.0 (mobile det+rec)       →  plate text via legacy ocr() API
"""

import os
import re
from pathlib import Path

# Point paddlex to bundled offline models BEFORE any paddle import.
_MODEL_DIR = Path(__file__).parent / "models"
os.environ["PADDLE_PDX_MODEL_HOME"] = str(_MODEL_DIR / "paddleocr")

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline as hf_pipeline
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
COCO_PERSON = 0
COCO_MOTO   = 3
COCO_CONF = 0.30;  COCO_IOU  = 0.45
S1_CONF   = 0.344; S1_IOU    = 0.45
S3_CONF   = 0.25;  S3_IOU    = 0.60
S4_CONF   = 0.20
PERSON_BIKE_IOU_THRESH = 0.10
PERSON_BIKE_COL_MARGIN = 0.35
HEAD_CROP_FRACTION = 0.45
HEAD_CROP_MIN_PX   = 40
DEPTH_THRESHOLD    = 0.35
OCR_MIN_CONF       = 0.25


class TrafficViolationDetector:
    """
    Detects traffic violations on two-wheelers in a single RGB image.
    All models loaded once in __init__; predict() is fully stateless.
    """

    def __init__(self, model_dir: str = "./models"):
        md = Path(model_dir)

        # Ensure paddlex finds bundled offline models
        os.environ["PADDLE_PDX_MODEL_HOME"] = str(md / "paddleocr")

        # 1. Depth estimation — model stored as fp16 on disk (47 MB vs 95 MB),
        #    but loaded as fp32 at runtime for fast CPU inference.
        self.depth_estimator = hf_pipeline(
            "depth-estimation",
            model=str(md / "depth_anything_v2"),
            device=0 if torch.cuda.is_available() else -1,
            dtype=torch.float32,
        )

        # 2. YOLO models
        self.s_coco = YOLO(str(md / "yolov8s.pt"))
        self.s1     = YOLO(str(md / "stage1_best.pt"))
        self.s3     = YOLO(str(md / "helmet_v11.pt"))
        self.s4     = YOLO(str(md / "license.pt"))

        # 3. Super-resolution (optional — falls back gracefully if missing)
        self.sr_engine, self.has_sr = self._init_sr(md / "FSRCNN_x3.pb")

        # 4. PaddleOCR 3.5.0 — mobile det + rec pipeline.
        #    Uses PP-OCRv5_mobile_det (4.7 MB) + en_PP-OCRv5_mobile_rec (7.6 MB).
        #    IMPORTANT: Must use the legacy .ocr() API, NOT .predict().
        #    The .predict() path triggers an OneDNN fused_conv2d crash on Windows,
        #    but .ocr() uses a compatible inference path that works everywhere.
        self.ocr_engine = PaddleOCR(
            lang="en",
            device="cpu",
            enable_mkldnn=False,
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="en_PP-OCRv5_mobile_rec",
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _init_sr(sr_path):
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
        except AttributeError:
            return None, False
        if Path(sr_path).exists():
            try:
                sr.readModel(str(sr_path))
                sr.setModel("fsrcnn", 3)
                return sr, True
            except Exception:
                pass
        return sr, False

    @staticmethod
    def _box_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0:
            return 0.0
        return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-6)

    @staticmethod
    def _region_depth(depth_map, x1, y1, x2, y2):
        h, w = depth_map.shape
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        patch = depth_map[y1:y2, x1:x2]
        return float(np.median(patch)) if patch.size > 0 else 0.5

    def _is_depth_ok(self, pd, bd):
        if bd < 0.05:
            return abs(pd - bd) <= DEPTH_THRESHOLD * 0.5
        return abs(pd - bd) / (bd + 1e-6) <= DEPTH_THRESHOLD

    def _merge_bike_boxes(self, coco, custom, iou_thresh=0.45):
        if not coco and not custom:
            return np.zeros((0, 4), dtype=np.float32)
        if not coco:
            return np.array(custom, dtype=np.float32)
        if not custom:
            return np.array(coco, dtype=np.float32)
        merged = list(coco)
        for cb in custom:
            if not any(self._box_iou(cb, mb) > iou_thresh for mb in merged):
                merged.append(cb)
        return np.array(merged, dtype=np.float32)

    def _associate_persons_to_bikes(self, person_boxes, bike_boxes, depth_map, h, w):
        bike_persons = [[] for _ in range(len(bike_boxes))]
        for p_box in person_boxes:
            px1, py1, px2, py2 = p_box
            p_cx = (px1 + px2) / 2
            p_bottom = py2
            best_bike, best_score = -1, -1.0
            for b_idx, b_box in enumerate(bike_boxes):
                bx1, by1, bx2, by2 = b_box
                bw = bx2 - bx1
                iou = self._box_iou(p_box, b_box)
                in_col = (
                    bx1 - PERSON_BIKE_COL_MARGIN * bw <= p_cx <= bx2 + PERSON_BIKE_COL_MARGIN * bw
                    and p_bottom <= by2 + 0.3 * (by2 - by1)
                )
                if iou < PERSON_BIKE_IOU_THRESH and not in_col:
                    continue
                pd_val = self._region_depth(depth_map, px1, py1, px2, py2)
                bd_val = self._region_depth(depth_map, bx1, by1, bx2, by2)
                if not self._is_depth_ok(pd_val, bd_val):
                    continue
                score = iou + 0.5 * (1.0 - abs(p_cx - (bx1 + bx2) / 2) / (w + 1e-6))
                if score > best_score:
                    best_score, best_bike = score, b_idx
            if best_bike >= 0:
                bike_persons[best_bike].append(p_box)
        return bike_persons

    def _get_depth_map(self, image_cv):
        img_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        result  = self.depth_estimator(Image.fromarray(img_rgb))
        depth   = np.array(result["depth"]).astype(np.float32)
        lo, hi  = depth.min(), depth.max()
        depth   = (depth - lo) / (hi - lo + 1e-8)
        if depth.shape != image_cv.shape[:2]:
            depth = cv2.resize(depth, (image_cv.shape[1], image_cv.shape[0]))
        return depth

    def _classify_helmets(self, full_image, person_boxes):
        if not person_boxes:
            return 0, 0, 0
        h_img, w_img = full_image.shape[:2]
        with_h = without_h = 0
        for p_box in person_boxes:
            px1, py1, px2, py2 = map(int, p_box)
            head_h = max(int((py2 - py1) * HEAD_CROP_FRACTION), HEAD_CROP_MIN_PX)
            pad_x  = max(4, int((px2 - px1) * 0.05))
            crop = full_image[max(0, py1):min(h_img, py1 + head_h),
                              max(0, px1 - pad_x):min(w_img, px2 + pad_x)]
            if crop.size == 0:
                without_h += 1
                continue
            res = self.s3.predict(crop, conf=S3_CONF, iou=S3_IOU, verbose=False)[0]
            if len(res.boxes) == 0:
                without_h += 1
            elif int(res.boxes[res.boxes.conf.argmax()].cls) == 0:
                with_h += 1
            else:
                without_h += 1
        return with_h + without_h, with_h, without_h

    def _preprocess_plate(self, plate_img):
        """Upscale and sharpen plate crop before OCR."""
        h, w = plate_img.shape[:2]
        if self.has_sr and self.sr_engine is not None:
            try:
                plate_img = self.sr_engine.upsample(plate_img)
            except Exception:
                plate_img = cv2.resize(plate_img, (0, 0), fx=3, fy=3,
                                       interpolation=cv2.INTER_CUBIC)
        else:
            if h < 100:
                scale = 100 / h
                plate_img = cv2.resize(plate_img,
                                       (int(w * scale), int(h * scale)),
                                       interpolation=cv2.INTER_CUBIC)
        lab = cv2.cvtColor(plate_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(l)
        plate_img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return cv2.filter2D(plate_img, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

    def _run_ocr(self, plate_img):
        """
        Full det+rec OCR on the plate crop using the legacy .ocr() API.

        PaddleOCR 3.5.0's .ocr() wraps .predict() but uses a compatible
        inference path that works on both Windows and Linux.
        The result is a list of dicts with 'rec_texts' and 'rec_scores' keys.
        """
        processed = self._preprocess_plate(plate_img)
        texts, scores = [], []
        try:
            result = self.ocr_engine.ocr(processed)
            if result and isinstance(result, list):
                for page in result:
                    if isinstance(page, dict):
                        # paddleocr 3.5.0 format: dict with rec_texts/rec_scores
                        page_texts  = page.get("rec_texts", [])
                        page_scores = page.get("rec_scores", [])
                        for t, s in zip(page_texts, page_scores):
                            if str(t).strip():
                                texts.append(str(t).strip())
                                scores.append(float(s))
                    elif isinstance(page, list):
                        # Legacy format: [[box, (text, score)], ...]
                        for line in page:
                            if isinstance(line, (list, tuple)) and len(line) == 2:
                                try:
                                    txt   = str(line[1][0])
                                    score = float(line[1][1])
                                    if txt.strip():
                                        texts.append(txt.strip())
                                        scores.append(score)
                                except (TypeError, ValueError, IndexError):
                                    pass
        except Exception:
            pass
        if not texts:
            return "UNKNOWN", 0.0
        return " ".join(texts), (sum(scores) / len(scores) if scores else 0.0)

    def _extract_plate(self, vehicle_crop, plate_box):
        """Crop plate from vehicle ROI, run OCR, return cleaned text."""
        h, w = vehicle_crop.shape[:2]
        pad = 4
        x1 = max(0, int(plate_box[0]) - pad)
        y1 = max(0, int(plate_box[1]) - pad)
        x2 = min(w, int(plate_box[2]) + pad)
        y2 = min(h, int(plate_box[3]) + pad)
        crop = vehicle_crop[y1:y2, x1:x2]
        if crop.size == 0:
            return "UNKNOWN"
        raw, conf = self._run_ocr(crop)
        if conf < OCR_MIN_CONF:
            return "UNKNOWN"
        text   = re.sub(r"[^A-Z0-9 \-]", "", raw.upper())
        text   = re.sub(r"\s+", " ", text).strip()
        tokens = [t for t in text.split() if len(t) > 1]
        return " ".join(tokens) if tokens else "UNKNOWN"

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, image_path: str) -> dict:
        """
        Run the full violation-detection pipeline on one image.

        Returns:
            {
                "violations": [
                    {
                        "num_riders":        int,
                        "helmet_violations": int,
                        "license_plate":     str
                    },
                    ...   # one entry per violating two-wheeler only
                ]
            }
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return {"violations": []}
            h_img, w_img = img.shape[:2]

            # Stage 1: COCO primary detection
            coco_res   = self.s_coco.predict(img, conf=COCO_CONF, iou=COCO_IOU,
                                             verbose=False)[0]
            coco_boxes = coco_res.boxes.xyxy.cpu().numpy()
            coco_cls   = coco_res.boxes.cls.cpu().numpy().astype(int)
            person_boxes = coco_boxes[coco_cls == COCO_PERSON].tolist()
            coco_motos   = coco_boxes[coco_cls == COCO_MOTO].tolist()

            # Stage 2: Supplemental bike detector
            s1_res       = self.s1.predict(img, conf=S1_CONF, iou=S1_IOU,
                                           augment=True, verbose=False)[0]
            custom_bikes = s1_res.boxes.xyxy.cpu().numpy().tolist()
            bike_boxes   = self._merge_bike_boxes(coco_motos, custom_bikes)
            if len(bike_boxes) == 0:
                return {"violations": []}

            # Stage 3: Depth map for spatial person→bike association
            depth_map = self._get_depth_map(img)

            # Stage 4: Associate persons to bikes
            bike_persons = self._associate_persons_to_bikes(
                person_boxes, bike_boxes, depth_map, h_img, w_img)

            # Stage 5-7: Per-bike helmet + plate + violation logic
            violations = []
            for i, bike_box in enumerate(bike_boxes):
                x1, y1, x2, y2 = map(int, bike_box)
                num_riders, with_h, without_h = self._classify_helmets(
                    img, bike_persons[i])

                # Fallback: no rider detected via COCO → assume 1 unclassified
                if num_riders == 0:
                    num_riders, with_h, without_h = 1, 0, 1

                # Expand bike box slightly to capture plate at bottom
                bw, bh = x2 - x1, y2 - y1
                vcrop = img[
                    max(0,     int(y1 - 0.20 * bh)): min(h_img, int(y2 + 0.10 * bh)),
                    max(0,     int(x1 - 0.15 * bw)): min(w_img, int(x2 + 0.15 * bw))
                ]

                plate_text = "UNKNOWN"
                try:
                    if vcrop.size > 0:
                        p_res = self.s4.predict(vcrop, conf=S4_CONF,
                                                verbose=False)[0]
                        if len(p_res.boxes) > 0:
                            best_pb = p_res.boxes.xyxy.cpu().numpy()[
                                p_res.boxes.conf.argmax()]
                            plate_text = self._extract_plate(vcrop, best_pb)
                except Exception:
                    plate_text = "UNKNOWN"

                # Violation: ≥3 riders OR any rider without helmet
                if (num_riders >= 3) or (without_h > 0):
                    violations.append({
                        "num_riders":        num_riders,
                        "helmet_violations": without_h,
                        "license_plate":     plate_text,
                    })

            return {"violations": violations}

        except Exception as e:
            print(f"[ERROR] predict() failed for {image_path}: {e}")
            return {"violations": []}
