import cv2
import json
import time
from pathlib import Path
from solution import TrafficViolationDetector

def annotate_image(img_path, result, out_path):
    img = cv2.imread(img_path)
    if img is None: return
    
    # Draw a semi-transparent overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (600, 50 + len(result['violations'])*40), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    # Write JSON results
    cv2.putText(img, 'EVALUATOR RESULTS:', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    y = 80
    for i, v in enumerate(result.get('violations', [])):
        text = f"[{i+1}] Riders: {v.get('num_riders')}, Helmet Viol: {v.get('helmet_violations')}, Plate: {v.get('license_plate')}"
        cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 40
        
    cv2.imwrite(out_path, img)

def main():
    print('--- Evaluator Simulation for baseline_alternative ---')
    print('1. Loading Detector (Loading Depth-Anything, YOLO, PaddleOCR)...')
    t0 = time.time()
    detector = TrafficViolationDetector(model_dir='./models')
    print(f'   -> Init took {time.time() - t0:.2f}s\n')
    
    # Test images
    test_dir = Path('../new_approach/final_results/test_2_v2')
    out_dir = Path('./output')
    out_dir.mkdir(exist_ok=True)
    
    images = list(test_dir.glob('*.jpg'))[:2] # Just test a couple images
    
    for img_path in images:
        print(f'-> Processing: {img_path.name}')
        t0 = time.time()
        result = detector.predict(str(img_path))
        elapsed = time.time() - t0
        
        # Save JSON
        json_path = out_dir / f"{img_path.stem}.json"
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=4)
            
        # Save Annotated Image
        annot_path = out_dir / f"annotated_{img_path.name}"
        annotate_image(str(img_path), result, str(annot_path))
        
        print(f'   Time: {elapsed:.2f}s')
        print(f'   JSON: {json.dumps(result)}')
        print(f'   Saved JSON: {json_path.name}')
        print(f'   Saved Image: {annot_path.name}\n')

if __name__ == '__main__':
    main()
