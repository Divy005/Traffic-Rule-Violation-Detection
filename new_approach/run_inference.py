"""
run_inference.py — Batch Inference + Annotation
=================================================
Runs the improved TrafficViolationDetector on all test images and saves:
  • output/annotated_<name>.jpg  — richly annotated image
  • output/<name>_result.json    — spec-compliant JSON

Usage:
    python run_inference.py
    python run_inference.py --src path/to/folder --model_dir path/to/models
"""

import sys, argparse, json, time, cv2, numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── colour palette ────────────────────────────────────────────────────────────
C_VEHICLE_OK  = (50,  205, 50)    # green   — compliant vehicle
C_VEHICLE_VIO = (0,   60,  255)   # red     — violating vehicle
C_RIDER       = (255, 165,  0)    # orange  — rider box
C_HELMET_OK   = (50,  205, 50)    # green   — helmet detected
C_HELMET_NO   = (0,   60,  255)   # red     — no helmet
C_HELMET_UNK  = (128, 128, 128)   # grey    — unknown
C_PLATE       = (0,   220, 255)   # cyan    — license plate
C_TEXT_BG     = (20,   20,  20)   # near-black label background
C_WHITE       = (255, 255, 255)


def _label(vis, text, x, y, fg, bg=C_TEXT_BG, scale=0.52, thick=1):
    """Draw a filled-background label."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(vis, (x, y - th - 4), (x + tw + 6, y + bl), bg, -1)
    cv2.putText(vis, text, (x + 3, y), font, scale, fg, thick, cv2.LINE_AA)


def draw_result(frame: np.ndarray, vehicle_results: list, result: dict) -> np.ndarray:
    """
    Rich annotation:
      • Vehicle bbox  → green (OK) / red (violation)
      • TRIPLE RIDING banner when num_riders > 2
      • Rider bboxes  → orange
      • Helmet bbox   → green / red / grey
      • Plate bbox    → cyan + OCR text
    """
    vis = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for veh in vehicle_results:
        vx1, vy1, vx2, vy2 = veh["vehicle_bbox"]
        nr    = veh["num_riders"]
        hviol = sum(1 for r in veh["riders"] if r.get("helmet") == "no_helmet")
        is_vio = (nr > 2 or hviol > 0)
        v_col = C_VEHICLE_VIO if is_vio else C_VEHICLE_OK

        # ── Vehicle box ───────────────────────────────────────────────────────
        cv2.rectangle(vis, (vx1, vy1), (vx2, vy2), v_col, 2)

        # ── Violation banners ─────────────────────────────────────────────────
        if nr > 2:
            banner = f"!! TRIPLE RIDING ({nr} riders) !!"
            (bw, bh), _ = cv2.getTextSize(banner, font, 0.72, 2)
            bx = max(0, vx1)
            by = max(bh + 8, vy1 - 4)
            cv2.rectangle(vis, (bx, by - bh - 6), (bx + bw + 10, by + 2), C_VEHICLE_VIO, -1)
            cv2.putText(vis, banner, (bx + 5, by - 2), font, 0.72, C_WHITE, 2, cv2.LINE_AA)
        elif hviol > 0:
            banner = f"!! HELMET VIOLATION ({hviol}/{nr}) !!"
            (bw, bh), _ = cv2.getTextSize(banner, font, 0.65, 2)
            bx = max(0, vx1)
            by = max(bh + 8, vy1 - 4)
            cv2.rectangle(vis, (bx, by - bh - 6), (bx + bw + 10, by + 2), C_VEHICLE_VIO, -1)
            cv2.putText(vis, banner, (bx + 5, by - 2), font, 0.65, C_WHITE, 2, cv2.LINE_AA)

        # ── Rider boxes ───────────────────────────────────────────────────────
        for ri, rider in enumerate(veh["riders"]):
            rx1, ry1, rx2, ry2 = rider["bbox"]
            cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), C_RIDER, 2)
            _label(vis, f"Rider {ri+1}", rx1, ry1 - 2, C_RIDER)

            # Helmet box
            h_label = rider.get("helmet", "unknown")
            h_bbox  = rider.get("helmet_bbox")
            h_conf  = rider.get("helmet_conf", 0.0)
            if h_label == "helmet":
                hcol = C_HELMET_OK
            elif h_label == "no_helmet":
                hcol = C_HELMET_NO
            else:
                hcol = C_HELMET_UNK

            if h_bbox:
                hx1, hy1, hx2, hy2 = h_bbox
                cv2.rectangle(vis, (hx1, hy1), (hx2, hy2), hcol, 2)
                _label(vis, f"{h_label} {h_conf:.2f}", hx1, hy2 + 14, hcol)
            else:
                # No bbox — just annotate inside rider box
                _label(vis, h_label, rx1 + 2, ry2 - 4, hcol)

        # ── License plate box + OCR ───────────────────────────────────────────
        p_bbox = veh.get("plate_bbox")
        lp_text = veh.get("license_plate") or ""
        if p_bbox:
            px1, py1, px2, py2 = p_bbox
            cv2.rectangle(vis, (px1, py1), (px2, py2), C_PLATE, 2)
            plate_label = f"LP: {lp_text}" if lp_text else "LP: ---"
            _label(vis, plate_label, px1, py1 - 2, C_PLATE, scale=0.60)

        # ── Bottom summary tag ────────────────────────────────────────────────
        lp_disp = lp_text or "-"
        summary = f"R:{nr}  H-viol:{hviol}  LP:{lp_disp}"
        _label(vis, summary, vx1, vy2 + 18, v_col, scale=0.50)

    # ── HUD (top-left) ────────────────────────────────────────────────────────
    violations_list = result.get("violations", [])
    total_vio = len(violations_list)
    hud_lines = [
        f"Vehicles  : {len(vehicle_results)}",
        f"Violations: {total_vio}",
    ]
    for i, line in enumerate(hud_lines):
        y = 28 + i * 24
        cv2.putText(vis, line, (8, y), font, 0.65, C_WHITE, 2, cv2.LINE_AA)
        cv2.putText(vis, line, (8, y), font, 0.65, (20, 20, 20), 1, cv2.LINE_AA)

    return vis


def process_image(detector, img_path: Path, out_dir: Path, verbose=True):
    """Run detector on one image, save annotated output + JSON."""
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"  [SKIP] Cannot read: {img_path.name}")
        return None

    t0 = time.perf_counter()

    # ── Run detector ──────────────────────────────────────────────────────────
    result = detector.predict(str(img_path))

    # Pull internal vehicle results for visualisation
    # (re-run internal stages to get full data — or rebuild from predict internals)
    vehicle_results = _get_vehicle_results(detector, frame)

    elapsed = round((time.perf_counter() - t0) * 1000, 1)

    # ── Annotate ──────────────────────────────────────────────────────────────
    vis = draw_result(frame, vehicle_results, result)

    # ── Save annotated image ──────────────────────────────────────────────────
    out_img = out_dir / f"annotated_{img_path.stem}.jpg"
    cv2.imwrite(str(out_img), vis)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_json = out_dir / f"{img_path.stem}_result.json"
    out_json.write_text(
        json.dumps(result, indent=2, default=str),
        encoding="utf-8"
    )

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  Image     : {img_path.name}")
        print(f"  Time      : {elapsed} ms")
        print(f"  JSON out  :\n{json.dumps(result, indent=4, default=str)}")
        print(f"  Annotated : {out_img.name}")
        print(f"  JSON file : {out_json.name}")

    return result, vehicle_results, vis


def _get_vehicle_results(detector, frame):
    """
    Re-run the internal pipeline to get full vehicle_results for visualisation.
    (predict() returns only spec JSON; we need the richer internal state.)
    """
    vehicle_results = detector._detect_vehicles_and_riders(frame)
    detector._classify_helmets(frame, vehicle_results)
    detector._read_plates(frame, vehicle_results)
    return vehicle_results


def _parse():
    p = argparse.ArgumentParser(description="Batch inference — new_approach")
    p.add_argument("--src",       default="../test_images",
                   help="Path to image folder (default: ../test_images)")
    p.add_argument("--model_dir", default="./models",
                   help="Model directory (default: ./models)")
    p.add_argument("--out",       default="output",
                   help="Output directory (default: output/)")
    p.add_argument("--show",      action="store_true",
                   help="Show each annotated image in a window")
    return p.parse_args()


def main():
    args = _parse()
    src_dir = Path(args.src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in exts)
    if not images:
        print(f"No images found in {src_dir}")
        return

    print(f"[Runner] Loading models from: {args.model_dir}")
    from solution import TrafficViolationDetector
    detector = TrafficViolationDetector(model_dir=args.model_dir)

    print(f"[Runner] Processing {len(images)} image(s) → {out_dir}/\n")

    all_results = []
    for img_path in images:
        out = process_image(detector, img_path, out_dir)
        if out is not None:
            result, vehicle_results, vis = out
            all_results.append({"image": img_path.name, "result": result})
            if args.show:
                cv2.imshow(f"new_approach — {img_path.name}", vis)
                cv2.waitKey(0)

    if args.show:
        cv2.destroyAllWindows()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  SUMMARY — {len(images)} image(s) processed")
    print(f"{'='*55}")
    total_violations = sum(
        len(r["result"].get("violations", [])) for r in all_results
    )
    print(f"  Total violating vehicles detected : {total_violations}")
    print(f"  Outputs saved to                  : {out_dir.resolve()}")
    print(f"{'='*55}\n")

    # Save combined summary JSON
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(all_results, indent=2, default=str),
        encoding="utf-8"
    )
    print(f"  Summary JSON → {summary_path}")


if __name__ == "__main__":
    main()
