import os
from ultralytics import YOLO

def find_best_weight(start_path="runs/detect"):
    for root, dirs, files in os.walk(start_path):
        if "best.pt" in files:
            return os.path.join(root, "best.pt")
    raise FileNotFoundError("❌ best.pt not found in runs/detect. Please train a model first.")

def main():
    # Locate best.pt
    weight_path = find_best_weight()
    print(f"✅ Found model weights at: {weight_path}")

    # Load the trained model
    model = YOLO(weight_path)

    # Set your test image path (replace with your own if needed)
    test_image = "datasets/coco128/images/train2017/000000000009.jpg"

    if not os.path.exists(test_image):
        raise FileNotFoundError(f"❌ Test image not found at: {test_image}")

    # Run prediction
    results = model.predict(source=test_image, conf=0.25, save=True)

    # Show result
    results[0].show()
    print("✅ Inference complete. Results saved to:", results[0].save_dir)

if __name__ == "__main__":
    main()
