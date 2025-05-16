import os
from ultralytics import YOLO

def find_best_weight(start_path="runs/detect"):
    for root, dirs, files in os.walk(start_path):
        if "best.pt" in files:
            return os.path.join(root, "best.pt")
    raise FileNotFoundError("❌ best.pt not found in runs/detect. Please train a model first.")

def main():
    # Find and load the best weights
    weight_path = find_best_weight()
    print(f"✅ Found model weights at: {weight_path}")
    model = YOLO(weight_path)

    # 🔄 Use your own image here
    test_image = "WhatsApp Image 2025-05-16 at 22.57.39_e1d0509b.jpg"  # <-- Replace with your image file
    conf_threshold = 0.1  # Lower this if you're getting "no detections"

    if not os.path.exists(test_image):
        raise FileNotFoundError(f"❌ Test image not found at: {test_image}")

    # Run prediction
    results = model.predict(source=test_image, conf=conf_threshold, save=True)

    # Show result
    results[0].show()
    print(f"✅ Inference complete. Results saved to: {results[0].save_dir}")

if __name__ == "__main__":
    main()
