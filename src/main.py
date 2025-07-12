import importlib
import os
import sys

# Add repo root to path for relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import parse_args_from_yaml


def main():
    args = parse_args_from_yaml("config/inference_timm_xgboost.yaml")
    detector = initialize_detector(args)
    if detector.setup():
        detector.run_inference()
        print(f"Inference complete. Results saved in: {args.output_dir}")
        # Create overall previews
        detector.create_scene_previews_with_bbox(confidence_threshold=0.5, merged=True)
        # Create individual previews for each detection
        detector.save_positive_detection_tiles(
            confidence_threshold=0.5, merged=False, patch_size=256
        )


def initialize_detector(args):
    """Dynamically imports and initializes the detector class based on config."""
    try:
        # If no module path is provided, assume the detector is in src.eddy_detector
        if "." in args.detector_class_name:
            module_name, class_name = args.detector_class_name.rsplit(".", 1)
        else:
            module_name = "src.eddy_detector"
            class_name = args.detector_class_name

        DetectorClass = getattr(importlib.import_module(module_name), class_name)
        detector = DetectorClass(args)
        print(f"Successfully instantiated detector: {args.detector_class_name}")
    except (AttributeError, ImportError) as e:
        print(
            f"Error: Could not find or import the detector class '{args.detector_class_name}'. Check config.yaml."
        )
        print(f"Details: {e}")
        sys.exit(1)
    return detector

if __name__ == "__main__":
    main()
