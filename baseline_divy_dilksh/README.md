# Traffic Violation Detection Pipeline

**Team Members:**
| Name | Roll Number |
|---|---|
| Divy Dobariya | BT2024225 |
| Dilksh Sharma | BT2024253 |

---

## Overview
Automated, fully offline traffic violation detector for two-wheelers.  
Detects: **helmet non-compliance**, **triple riding (>2 riders)**, and **license plate extraction**.

## Folder Structure
```
baseline_divy_dilksh/
├── solution.py          ← TrafficViolationDetector class (evaluator entry point)
├── requirements.txt     ← Pinned dependency versions
├── report.tex           ← LaTeX source of project report
└── models/
    ├── yolov8s.pt                        (22.6 MB) — Vehicle + person detection
    ├── helmet_best.pt                    (22.5 MB) — Helmet classification
    ├── license-plate-finetune-v1x.pt    (114.4 MB) — Plate detection
    └── paddleocr_weights/                (15.8 MB) — Offline OCR weights
        ├── det/en/en_PP-OCRv3_det_infer/
        ├── rec/en/en_PP-OCRv4_rec_infer/
        └── cls/ch_ppocr_mobile_v2.0_cls_infer/

Total model size: ~175.3 MB  (limit: 250 MB ✓)
```

## How to Run (Evaluator Protocol)

```python
from solution import TrafficViolationDetector

detector = TrafficViolationDetector(model_dir="./models")
result   = detector.predict("path/to/image.jpg")
print(result)
```

### Expected Output Format
```json
{
    "violations": [
        {
            "num_riders": 2,
            "helmet_violations": 1,
            "license_plate": "DL5AE190"
        }
    ]
}
```
- Only **violating** vehicles appear (`helmet_violations > 0` OR `num_riders > 2`).
- `license_plate` is `null` if the plate cannot be read.

## Dependencies
Install with:
```bash
pip install -r requirements.txt
```

## Key Design Decisions
1. **Foot-zone spatial filter** — pseudo-depth cue prevents pedestrians from being counted as riders.
2. **Head-ROI helmet classification** — top 42% of rider crop reduces body/background noise.
3. **CLAHE + Unsharp Mask preprocessing** — improves OCR on low-contrast plates.
4. **IND/ND hologram artifact stripping** — fixes common Indian plate OCR errors.
5. **Fully offline** — all PaddleOCR weights bundled in `models/paddleocr_weights/`.
