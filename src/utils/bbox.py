import math
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from shapely import affinity
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry


def merge_csv_bboxes(
    csv_path: str,
    nms_iou_threshold: float = 0.05,
    merge_iou_threshold: float = 0.3,
    post_nms_iou_threshold: float = 0.1,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Three-stage postprocessing of eddy detections from a CSV file.

    This function implements:
      1) Non-Maximum Suppression (NMS) to prune overlapping detection windows.
         - NMS is an algorithm that selects only the most confident bounding boxes
           among a set of overlapping boxes.
         - It relies on the Intersection over Union (IoU) metric, defined as:
           IoU(A, B) = area(A ∩ B) / area(A ∪ B).
         - NMS iteratively removes lower scoring boxes which have an IoU greater
           than iou_threshold with another (higher scoring) box.
         - Lower thresholds (e.g. 0.2) aggressively collapse near-duplicates;
           higher thresholds (e.g. 0.5+) keep more overlapping boxes.
      2) Union-based merging of surviving boxes based on a merge IoU threshold.
         - Any two boxes whose IoU > merge threshold are merged into a single
           polygon (their geometric union).
      3) Post-merge NMS to remove any redundant merged extents.
         - Same greedy IoU logic as step 1, but applied to the merged boxes.

    Geospatial considerations:
      - Longitude degrees correspond to smaller ground distances near the poles.
        To approximate true (geodesic) areas, we scale the longitude dimension by
        cos(mean_latitude) before computing areas.
      - This "latitude-aware" scaling ensures that IoU comparisons remain meaningful
        across different latitudes.

    Configuration Integration:
      - IoU thresholds are configurable through the Hydra configuration system.
      - Default values are defined in config/inference/default.yaml under bbox_merging.
      - Parameters can be overridden via CLI:
        python src/main.py inference.bbox_merging.nms_iou_threshold=0.1
      - The three thresholds control different stages of the merging pipeline:
        * nms_iou_threshold: Initial suppression of overlapping detections
        * merge_iou_threshold: Union-based combining of similar boxes
        * post_nms_iou_threshold: Final cleanup of merged results

    Args:
        csv_path (str): Path to a CSV file with columns:
            - 'filename': identifier for each image or tile
            - 'bbox': string representation of bounding box [xmin ymin xmax ymax]
            - 'confidence': float confidence score for each detection
        nms_iou_threshold (float): IoU threshold for initial NMS stage. Default 0.05.
        merge_iou_threshold (float): IoU threshold for union-based merging. Default 0.3.
        post_nms_iou_threshold (float): IoU threshold for post-merge NMS. Default 0.1.

        Class-awareness:
            - If the CSV includes a 'pred_class' column (and optional 'pred_label'),
                merging is performed per class to avoid fusing different classes.
            - Output dictionaries will include 'pred_class' and 'pred_label' when
                present in the input, allowing downstream consumers to persist class info.

        Returns:
                Dict[str, List[Dict[str, Any]]]:
                        Mapping from filename to a list of output detections, each with:
                            - 'bbox': Tuple[float, float, float, float] for (xmin, ymin, xmax, ymax)
                            - 'confidence': float averaged over any merged detections
                            - Optional 'pred_class' (int) and 'pred_label' (str)
    """
    # Attempt to read the CSV; return empty results on failure
    try:
        df = pd.read_csv(csv_path)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return {}

    # Validate required columns
    if df.empty or not {"filename", "bbox", "confidence"}.issubset(df.columns):
        return {}

    def to_shapely(b: str) -> Optional[BaseGeometry]:
        """Helper: convert a bbox string to a shapely box, or None if invalid."""
        parsed = parse_bbox(b)
        return box(*parsed) if parsed is not None else None

    # Parse each bbox into a shapely geometry for spatial operations
    df["shape"] = df["bbox"].apply(to_shapely)
    df = df[df["shape"].notnull()]
    # Normalize class fields if present
    has_class = "pred_class" in df.columns
    has_label = "pred_label" in df.columns
    if has_class:
        # Coerce to integers when possible
        df["pred_class"] = pd.to_numeric(df["pred_class"], errors="coerce").astype(
            "Int64"
        )
        df = df[df["pred_class"].notnull()]

    results: Dict[str, List[Dict[str, Any]]] = {}

    # Process each image/tile separately
    for filename, group in df.groupby("filename"):
        out_detections: List[Dict[str, Any]] = []

        # If class column exists, operate per class to avoid cross-class merges
        class_groups = group.groupby("pred_class") if has_class else [(None, group)]

        for class_id, cls_group in class_groups:
            # Determine a representative label for this class_id if present
            class_label: Optional[str] = None
            if has_label and not cls_group["pred_label"].empty:
                try:
                    # Use the most frequent label; fallback to first
                    class_label = (
                        cls_group["pred_label"].mode().iloc[0]
                        if not cls_group["pred_label"].mode().empty
                        else str(cls_group["pred_label"].iloc[0])
                    )
                except Exception:
                    class_label = str(cls_group["pred_label"].iloc[0])

            # Stage 1: Pre-merge NMS to suppress redundant windows
            shapes = cast(List[BaseGeometry], cls_group["shape"].tolist())
            scores = cls_group["confidence"].to_numpy()
            keep_indices = _nms_shapely(shapes, scores, nms_iou_threshold)
            survivors = cls_group.iloc[keep_indices]

            # Stage 2: Union-based merging of boxes whose IoU exceeds merge_iou
            detections: List[Dict[str, Any]] = []
            for _, row in survivors.iterrows():
                det: Dict[str, Any] = {
                    "shape": row["shape"],
                    "confidences": [row["confidence"]],
                }
                if has_class:
                    det["pred_class"] = int(class_id)  # type: ignore[arg-type]
                if has_label and class_label is not None:
                    det["pred_label"] = class_label
                detections.append(det)

            merged = _union_merge_boxes(detections, merge_iou_threshold)

            # Stage 3: Post-merge NMS to clean up any overlapping merged extents
            if merged:
                merged_shapes: List[BaseGeometry] = [box(*d["bbox"]) for d in merged]
                merged_scores = np.array([d["confidence"] for d in merged])
                final_keep = _nms_shapely(
                    merged_shapes, merged_scores, post_nms_iou_threshold
                )
                merged = [merged[i] for i in final_keep]

            out_detections.extend(merged)

    results[str(filename)] = out_detections

    return results


def parse_bbox(bbox_str: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Parses a string representation of a bounding box into numerical coordinates.

    Expected format: "[xmin ymin xmax ymax]" or "xmin ymin xmax ymax".

    Args:
        bbox_str (str): String to parse.

    Returns:
        Optional[Tuple[float, float, float, float]]:
            Tuple of floats (xmin, ymin, xmax, ymax) if parsing succeeds;
            otherwise None.
    """
    try:
        parts = bbox_str.strip("[]").split()
        if len(parts) != 4:
            return None
        coords = tuple(map(float, parts))
        return coords
    except (ValueError, AttributeError):
        return None


def _nms_shapely(
    boxes: List[BaseGeometry], scores: np.ndarray, iou_threshold: float
) -> List[int]:
    """
    Latitude-aware Non-Maximum Suppression (NMS).

    NMS selects a subset of boxes by iteratively picking the box with the highest score
    and removing all boxes whose IoU with it exceeds the threshold.

    Args:
        boxes (List[BaseGeometry]): List of shapely geometries representing bboxes.
        scores (np.ndarray): Confidence scores corresponding to each box.
        iou_threshold (float): IoU cutoff above which boxes are suppressed.

    Returns:
        List[int]: Indices of boxes that are kept after suppression.
    """
    if not boxes:
        return []
    # Sort boxes by descending score
    order = scores.argsort()[::-1]
    keep: List[int] = []
    # Iterate until no boxes remain
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        # Compute IoUs between the chosen box and the rest
        others = order[1:]
        ious = np.array(
            [_calculate_latitude_aware_iou(boxes[i], boxes[j]) for j in others]
        )
        # Keep boxes whose IoU is below threshold
        below_threshold = np.where(ious <= iou_threshold)[0]
        order = order[below_threshold + 1]
    return keep


def _union_merge_boxes(
    detections: List[Dict[str, Any]], iou_threshold: float
) -> List[Dict[str, Any]]:
    """
    Iterative union-based merging of detections.

    For each pair of detections, if their IoU exceeds iou_threshold, merge them into one
    by taking the geometric union and combining their confidence lists. Repeat until stable.

    Args:
        detections (List[Dict[str, Any]]): Each dict has:
            - 'shape': BaseGeometry for the box
            - 'confidences': List of float scores
        iou_threshold (float): IoU cutoff above which boxes are merged.

    Returns:
        List[Dict[str, Any]]: Merged detections, each with:
            - 'bbox': Tuple[float, float, float, float]
            - 'confidence': float average of merged confidences
            - Optional: 'pred_class' and 'pred_label' when provided
    """
    has_merged = True
    while has_merged:
        has_merged = False
        i = 0
        # Compare each pair in sequence
        while i < len(detections):
            j = i + 1
            while j < len(detections):
                iou_val = _calculate_latitude_aware_iou(
                    detections[i]["shape"], detections[j]["shape"]
                )
                if iou_val > iou_threshold:
                    # Merge shapes and aggregate confidences
                    detections[i]["shape"] = detections[i]["shape"].union(
                        detections[j]["shape"]
                    )
                    detections[i]["confidences"].extend(detections[j]["confidences"])
                    detections.pop(j)
                    has_merged = True
                else:
                    j += 1
            i += 1
    # Format output with averaged confidence per merged shape
    merged_out: List[Dict[str, Any]] = []
    for det in detections:
        out: Dict[str, Any] = {
            "bbox": det["shape"].bounds,  # (xmin, ymin, xmax, ymax)
            "confidence": float(np.mean(det["confidences"])),
        }
        # Preserve class metadata when present
        if "pred_class" in det:
            out["pred_class"] = det["pred_class"]
        if "pred_label" in det:
            out["pred_label"] = det["pred_label"]
        merged_out.append(out)
    return merged_out


def _calculate_latitude_aware_iou(box1: BaseGeometry, box2: BaseGeometry) -> float:
    """
    Calculates latitude-aware Intersection over Union (IoU) for two boxes.

    To account for the Earth's curvature, we scale the longitude dimension by
    cos(mean_latitude) before computing areas.

    Args:
        box1 (BaseGeometry): First bounding box geometry.
        box2 (BaseGeometry): Second bounding box geometry.

    Returns:
        float: Scaled IoU between the two boxes in [0.0, 1.0].
    """
    # Compute mean latitude of the two box centroids
    mean_lat = (box1.centroid.y + box2.centroid.y) / 2.0
    # Scale factor for longitude distances
    scale = math.cos(math.radians(mean_lat))

    # Scale boxes in the longitude (x) direction to approximate true geo-area
    scaled1 = affinity.scale(box1, xfact=scale, yfact=1.0, origin="center")
    scaled2 = affinity.scale(box2, xfact=scale, yfact=1.0, origin="center")

    inter_area = scaled1.intersection(scaled2).area
    union_area = scaled1.union(scaled2).area
    return inter_area / union_area if union_area > 0 else 0.0

