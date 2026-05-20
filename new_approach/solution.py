"""
solution.py — Improved Traffic Violation Detection
====================================================
new_approach/ — Divy Dobariya | BT2024225

Improvements over BT2024225/solution.py:
  1. Head-only ROI for helmet classification (top 40% of rider crop)
  2. Height-consistency filter to remove pedestrian false positives
  3. Prefer license-plate-finetune-v1x.pt (better plate recall)
  4. CLAHE + sharpening preprocessing before OCR
  5. Indian plate regex with OCR char-error correction
  6. Distance-based plate fallback when no overlap found
  7. Triple-riding clearly captured in num_riders field

PDF Output spec:
{
  "violations": [
    {
      "num_riders":        int,
      "helmet_violations": int,
      "license_plate":     "string | null"
    }
  ]
}
Only violating vehicles (helmet_violations > 0 OR num_riders > 2) appear.
"""

import re
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ─── COCO class IDs ──────────────────────────────────────────────────────────
PERSON_CLASS     = 0
MOTORCYCLE_CLASS = 3

# ─── Detection thresholds ────────────────────────────────────────────────────
DET_CONF      = 0.25
DET_IOU       = 0.45
PERSON_CONF   = 0.20   # low to catch partially-visible riders
HELMET_CONF   = 0.25
PLATE_CONF    = 0.12   # very low → maximise plate recall
PLATE_IOU     = 0.45

# ─── Spatial association ──────────────────────────────────────────────────────
EXPAND_UP     = 1.3    # riders sit above the bike
EXPAND_SIDES  = 0.15
EXPAND_DOWN   = 0.05
OVERLAP_THRESH = 0.20

# ─── Helmet: only inspect the head region ────────────────────────────────────
HEAD_RATIO    = 0.42   # top 42 % of rider crop height = head

# ─── OCR ─────────────────────────────────────────────────────────────────────
PLATE_PAD     = 8
MIN_PLATE_H   = 52     # upscale plate crops below this height

# Indian plate pattern: 2 letters, 2 digits, 1–3 letters, 4 digits
_PLATE_RE     = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$')

# Character confusion maps (position-aware correction)
_D2L = {'0': 'O', '1': 'I', '8': 'B', '5': 'S', '2': 'Z', '6': 'G'}  # digit→letter
_L2D = {'O': '0', 'I': '1', 'L': '1', 'B': '8', 'S': '5', 'Z': '2',
        'G': '6', 'Q': '0', 'D': '0'}                                   # letter→digit


# ─────────────────────────────────────────────────────────────────────────────
class TrafficViolationDetector:
    """
    Offline, stateless traffic violation detector.

    Usage (mirrors evaluator protocol):
        detector = TrafficViolationDetector(model_dir="./models")
        result   = detector.predict("image.jpg")
    """

    # ── Model discovery ───────────────────────────────────────────────────────
    _SEARCH_ROOTS = [
        "models",
        "../baseline_divy_dilksh/models",
        "../models",
    ]

    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)
        self._roots = [self.model_dir] + [Path(r) for r in self._SEARCH_ROOTS]

        # Primary detector — prefer yolov8m (better recall), fallback to s then n
        det_path = (self._find("yolov8m.pt")
                    or self._find("yolov8s.pt")
                    or self._find("yolov8n.pt"))
        if det_path is None:
            raise FileNotFoundError("yolov8s.pt or yolov8n.pt not found in search paths")
        print(f"[Init] Detector  : {det_path}")
        self.det_model = YOLO(det_path)

        # Helmet classifier
        hpath = self._find("helmet_best.pt")
        if hpath is None:
            raise FileNotFoundError("helmet_best.pt not found")
        print(f"[Init] Helmet    : {hpath}")
        self.helmet_model = YOLO(hpath)

        # Plate detector — prefer the v1x model (much better recall)
        ppath = (self._find("license-plate-finetune-v1x.pt")
                 or self._find("best (2).pt")
                 or self._find("best.pt"))
        if ppath is None:
            raise FileNotFoundError("No plate detection model found")
        print(f"[Init] Plate     : {ppath}")
        self.plate_model = YOLO(ppath)

        # PaddleOCR — prefer offline weights
        ocr_dir = self._find_dir("paddleocr_weights")
        if ocr_dir:
            print(f"[Init] PaddleOCR : {ocr_dir} (offline)")
            base = Path(ocr_dir)
            self.ocr = PaddleOCR(
                det_model_dir=str(base / "det" / "en" / "en_PP-OCRv3_det_infer"),
                rec_model_dir=str(base / "rec" / "en" / "en_PP-OCRv4_rec_infer"),
                cls_model_dir=str(base / "cls" / "ch_ppocr_mobile_v2.0_cls_infer"),
                use_angle_cls=True, lang="en", show_log=False,
            )
        else:
            print("[Init] PaddleOCR : online weights")
            self.ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

        print("[Init] All models ready.\n")

    def _find(self, name: str):
        for root in self._roots:
            p = root / name
            if p.exists():
                return str(p)
        return None

    def _find_dir(self, name: str):
        for root in self._roots:
            p = root / name
            if p.is_dir():
                return str(p)
        return None

    # ── Geometry helpers ──────────────────────────────────────────────────────
    def _expand(self, bbox, img_h, img_w):
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        return [
            max(0,     int(x1 - bw * EXPAND_SIDES)),
            max(0,     int(y1 - bh * EXPAND_UP)),
            min(img_w, int(x2 + bw * EXPAND_SIDES)),
            min(img_h, int(y2 + bh * EXPAND_DOWN)),
        ]

    def _overlap_frac(self, boxA, boxB):
        ax1, ay1, ax2, ay2 = boxA
        bx1, by1, bx2, by2 = boxB
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return (ix2 - ix1) * (iy2 - iy1) / max(1, (ax2 - ax1) * (ay2 - ay1))

    def _bottom_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) // 2, y2

    def _pt_in_box(self, pt, box):
        x, y = pt
        return box[0] <= x <= box[2] and box[1] <= y <= box[3]

    def _foot_zone(self, bbox, img_h, img_w):
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        # Tighter zone — only accept riders whose seat/feet are within the bike area
        return [
            max(0, int(x1 - bw * 0.15)),
            max(0, int(y1 - bh * 0.10)),   # was 0.20 → tighter upward bound
            min(img_w, int(x2 + bw * 0.15)),
            min(img_h, int(y2 + bh * 0.10)),
        ]

    def _nms_vehicles(self, vehicles, iou_thresh=0.50):
        """Suppress duplicate vehicle detections that heavily overlap each other."""
        if len(vehicles) <= 1:
            return vehicles
        # Sort by confidence descending
        vehicles = sorted(vehicles, key=lambda v: v["conf"], reverse=True)
        keep = []
        suppressed = set()
        for i, va in enumerate(vehicles):
            if i in suppressed:
                continue
            keep.append(va)
            ax1, ay1, ax2, ay2 = va["bbox"]
            a_area = max(1, (ax2-ax1)*(ay2-ay1))
            for j, vb in enumerate(vehicles[i+1:], i+1):
                if j in suppressed:
                    continue
                bx1, by1, bx2, by2 = vb["bbox"]
                ix1, iy1 = max(ax1,bx1), max(ay1,by1)
                ix2, iy2 = min(ax2,bx2), min(ay2,by2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2-ix1)*(iy2-iy1)
                    b_area = max(1, (bx2-bx1)*(by2-by1))
                    iou = inter / (a_area + b_area - inter)
                    if iou > iou_thresh:
                        suppressed.add(j)
        return keep

    # ── Stage 1 + 2: Vehicle & Rider Detection + Association ─────────────────
    def _detect_vehicles_and_riders(self, frame):
        img_h, img_w = frame.shape[:2]

        # Pass 1: motorcycles
        moto = self.det_model.predict(
            frame, conf=DET_CONF, iou=DET_IOU,
            classes=[MOTORCYCLE_CLASS], verbose=False
        )[0]
        vehicles = [
            {"bbox": list(map(int, b.xyxy[0].tolist())), "conf": float(b.conf.item())}
            for b in moto.boxes
        ]
        # ── NMS: remove duplicate/heavily-overlapping bike detections ─────────
        vehicles = self._nms_vehicles(vehicles, iou_thresh=0.50)

        # Pass 2: persons (lower conf = more recall)
        pers = self.det_model.predict(
            frame, conf=PERSON_CONF, iou=DET_IOU,
            classes=[PERSON_CLASS], verbose=False
        )[0]
        persons = [
            {"bbox": list(map(int, b.xyxy[0].tolist())), "conf": float(b.conf.item())}
            for b in pers.boxes
        ]

        # Fallback: no motorcycles detected → treat full image as one zone
        if not vehicles:
            if persons:
                vehicles = [{"bbox": [0, 0, img_w, img_h], "conf": 0.5}]
            else:
                return []

        # Spatial association
        foot_zones  = [self._foot_zone(v["bbox"], img_h, img_w) for v in vehicles]
        assignments = [-1] * len(persons)

        for pi, person in enumerate(persons):
            pbbox = person["bbox"]
            bc    = self._bottom_center(pbbox)
            best_v, best_score = -1, -1.0
            for vi, (veh, fz) in enumerate(zip(vehicles, foot_zones)):
                # Overlap with original vehicle bbox (not massively expanded)
                overlap = self._overlap_frac(pbbox, veh["bbox"])
                
                # Must be in foot zone AND have significant overlap with the vehicle
                if self._pt_in_box(bc, fz) and overlap >= OVERLAP_THRESH:
                    # Use overlap to break ties, with a tiny distance penalty
                    vx_c = (veh["bbox"][0] + veh["bbox"][2]) / 2
                    vy_c = (veh["bbox"][1] + veh["bbox"][3]) / 2
                    dist = ((bc[0] - vx_c)**2 + (bc[1] - vy_c)**2)**0.5
                    score = overlap - (dist / 10000.0)
                    
                    if score > best_score:
                        best_score, best_v = score, vi
            assignments[pi] = best_v

        # Build vehicle result dicts
        vehicle_results = []
        for vi, veh in enumerate(vehicles):
            riders = [persons[pi] for pi, av in enumerate(assignments) if av == vi]

            # ── Improvement 2: height-consistency filter ──────────────────
            # Removes bystanders whose size is an outlier vs the rest of the group
            if len(riders) >= 2:
                heights = sorted(r["bbox"][3] - r["bbox"][1] for r in riders)
                median_h = heights[len(heights) // 2]
                riders = [
                    r for r in riders
                    if abs((r["bbox"][3] - r["bbox"][1]) - median_h) / max(1, median_h) < 0.60
                ]

            vehicle_results.append({
                "vehicle_bbox":  veh["bbox"],
                "num_riders":    len(riders),
                "riders":        riders,
                "license_plate": None,
                "plate_bbox":    None,
            })

        return vehicle_results

    # ── Stage 3: Helmet Classification ───────────────────────────────────────
    def _head_crop(self, rider_bbox, img_h, img_w):
        """Return the head sub-region of a rider bbox (top HEAD_RATIO of height)."""
        x1, y1, x2, y2 = rider_bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        head_y2 = int(y1 + (y2 - y1) * HEAD_RATIO)
        head_y2 = max(head_y2, y1 + 16)   # ensure at least 16 px
        return x1, y1, x2, head_y2

    def _classify_helmets(self, frame, vehicle_results):
        img_h, img_w = frame.shape[:2]
        for veh in vehicle_results:
            for rider in veh["riders"]:
                # ── Improvement 1: run on head region first ───────────────
                hx1, hy1, hx2, hy2 = self._head_crop(rider["bbox"], img_h, img_w)
                crop = frame[hy1:hy2, hx1:hx2]

                res = None
                off_x, off_y = hx1, hy1

                if crop.size > 0:
                    res = self.helmet_model.predict(
                        crop, conf=HELMET_CONF, iou=0.45, verbose=False
                    )[0]

                # Fallback: if head crop gave no detection, try full rider body
                if res is None or len(res.boxes) == 0:
                    x1, y1, x2, y2 = rider["bbox"]
                    full = frame[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
                    if full.size > 0:
                        res = self.helmet_model.predict(
                            full, conf=HELMET_CONF, iou=0.45, verbose=False
                        )[0]
                        off_x, off_y = x1, y1

                if res is None or len(res.boxes) == 0:
                    rider["helmet"]      = "unknown"
                    rider["helmet_conf"] = 0.0
                    rider["helmet_bbox"] = None
                    continue

                best = max(res.boxes, key=lambda b: float(b.conf.item()))
                label = self.helmet_model.names[int(best.cls.item())]
                bx1, by1, bx2, by2 = map(int, best.xyxy[0].tolist())
                rider["helmet"]      = label
                rider["helmet_conf"] = round(float(best.conf.item()), 3)
                rider["helmet_bbox"] = [off_x + bx1, off_y + by1,
                                        off_x + bx2, off_y + by2]

    # ── OCR helpers ───────────────────────────────────────────────────────────
    def _preprocess_plate(self, crop):
        """CLAHE + sharpen + upscale for better PaddleOCR accuracy."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # CLAHE — handles low contrast / shadows
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)
        # Unsharp-mask sharpening
        blur = cv2.GaussianBlur(gray, (0, 0), 2)
        gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
        proc = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # Upscale if crop is too small for OCR
        h = proc.shape[0]
        if h < MIN_PLATE_H:
            scale = MIN_PLATE_H / h
            proc = cv2.resize(proc, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        return proc

    def _fix_plate_text(self, raw: str) -> str:
        """
        Position-aware OCR character correction for Indian plates.
        Plate format: LL DD L{1-3} DDDD  (L=letter, D=digit)
        """
        text = "".join(c for c in raw.upper() if c.isalnum())
        
        # Remove common "IND" or "ND" artifact
        text = text.replace("IND", "")
        match = re.match(r'^([A-Z0-9]{4})ND([A-Z0-9]{1,3}\d{4})$', text)
        if match:
            text = match.group(1) + match.group(2)
            
        if len(text) < 5:
            return text

        t = list(text)
        # Positions 0,1 → letters
        for i in [0, 1]:
            if i < len(t) and t[i] in _L2D:
                # digit where letter expected → convert back
                t[i] = _D2L.get(t[i], t[i])
        # Positions 2,3 → digits
        for i in [2, 3]:
            if i < len(t) and t[i] in _L2D:
                t[i] = _L2D[t[i]]
        # Middle letters (positions 4 to len-5) — letters expected
        for i in range(4, max(4, len(t) - 4)):
            if t[i] in _L2D:
                t[i] = _D2L.get(t[i], t[i])
        # Last 4 → digits
        for i in range(max(0, len(t) - 4), len(t)):
            if t[i] in _L2D:
                t[i] = _L2D[t[i]]

        fixed = "".join(t)
        return fixed if len(fixed) >= 5 else text

    # ── Stage 4: Plate Detection + OCR ───────────────────────────────────────
    def _read_plates(self, frame, vehicle_results):
        h_img, w_img = frame.shape[:2]

        plate_res = self.plate_model.predict(
            frame, conf=PLATE_CONF, iou=PLATE_IOU, verbose=False
        )[0]
        plates = [list(map(int, b.xyxy[0].tolist())) for b in plate_res.boxes]

        for veh in vehicle_results:
            vx1, vy1, vx2, vy2 = veh["vehicle_bbox"]
            # Extend vehicle zone downward a bit (plate can be just below bbox)
            vy2_ext = min(h_img, vy2 + int((vy2 - vy1) * 0.25))

            best_plate, best_score = None, 0

            for plate in plates:
                px1, py1, px2, py2 = plate
                # Overlap with extended vehicle zone
                ix1 = max(vx1, px1); iy1 = max(vy1, py1)
                ix2 = min(vx2, px2); iy2 = min(vy2_ext, py2)
                if ix1 < ix2 and iy1 < iy2:
                    score = (ix2 - ix1) * (iy2 - iy1)
                    if score > best_score:
                        best_score, best_plate = score, plate

            # Last resort: closest plate by centre distance
            if best_plate is None and plates:
                vc = ((vx1 + vx2) // 2, (vy1 + vy2) // 2)
                best_plate = min(
                    plates,
                    key=lambda p: (((p[0]+p[2])//2 - vc[0])**2 +
                                   ((p[1]+p[3])//2 - vc[1])**2)
                )

            if best_plate is None:
                continue

            veh["plate_bbox"] = best_plate
            px1, py1, px2, py2 = best_plate
            px1 = max(0,     px1 - PLATE_PAD)
            py1 = max(0,     py1 - PLATE_PAD)
            px2 = min(w_img, px2 + PLATE_PAD)
            py2 = min(h_img, py2 + PLATE_PAD)

            crop = frame[py1:py2, px1:px2]
            if crop.size == 0:
                continue

            proc = self._preprocess_plate(crop)
            text = self._ocr_with_fallback(proc)
            if text and len(text) >= 3:
                veh["license_plate"] = text

    def _ocr_with_fallback(self, proc):
        """Run PaddleOCR; if result is empty/short, retry with 90-deg rotation."""
        def _run_ocr(img):
            try:
                ocr_out = self.ocr.ocr(img, cls=True)
                if ocr_out and ocr_out[0]:
                    raw  = "".join(line[1][0] for line in ocr_out[0]).strip()
                    return self._fix_plate_text(raw)
            except Exception as e:
                print(f"  [OCR] Skipped: {e}")
            return ""

        text = _run_ocr(proc)
        if len(text) >= 5:
            return text
        # Retry with 90-degree clockwise rotation (handles sideways plates)
        rotated = cv2.rotate(proc, cv2.ROTATE_90_CLOCKWISE)
        text2 = _run_ocr(rotated)
        return text2 if len(text2) > len(text) else text

    # ── Public API ────────────────────────────────────────────────────────────
    def predict(self, image_path: str) -> dict:
        """
        Analyse one image and return spec-compliant violation dictionary.

        Returns:
            {
              "violations": [
                {
                  "num_riders":        int,   # total riders on this vehicle
                  "helmet_violations": int,   # count of riders without helmets
                  "license_plate":     str|None
                }
              ]
            }
        Only violating vehicles appear (num_riders > 2 OR helmet_violations > 0).
        """
        frame = cv2.imread(str(image_path))
        if frame is None:
            return {"violations": []}

        vehicle_results = self._detect_vehicles_and_riders(frame)
        self._classify_helmets(frame, vehicle_results)
        self._read_plates(frame, vehicle_results)

        violations_out = []
        for veh in vehicle_results:
            nr    = veh["num_riders"]
            hviol = sum(1 for r in veh["riders"] if r.get("helmet") == "no_helmet")
            if nr > 2 or hviol > 0:
                violations_out.append({
                    "num_riders":        nr,
                    "helmet_violations": hviol,
                    "license_plate":     veh.get("license_plate"),
                })

        return {"violations": violations_out}
