import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class DepthEstimator:
    """Depth Anything V2 based depth estimation for pothole severity analysis."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "depth-anything/Depth-Anything-V2-Small-hf"

        # Load Depth Anything V2 Small model
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Run Depth Anything V2 on a full frame and return a normalized depth map (0-1).

        Args:
            frame: RGB numpy array (H, W, 3)

        Returns:
            Normalized depth map (H, W) with values in [0, 1]
        """
        pil_image = Image.fromarray(frame)

        inputs = self.image_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Normalize to 0-1
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 0:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_map = np.zeros_like(depth_map)

        return depth_map

    def compute_severity_score(self, depth_map: np.ndarray, bbox: np.ndarray,
                                 image_shape: tuple) -> dict:
        """
        Compute a composite severity score using multiple signals:
          1. Depth contrast  - how much the pothole depth differs from road surface
          2. Depth variance  - rougher/deeper potholes have more depth variation
          3. Size factor     - larger potholes are more severe

        Args:
            depth_map: Normalized depth map (0-1)
            bbox: Bounding box [x1, y1, x2, y2]
            image_shape: (H, W) of the original image

        Returns:
            Dict with 'score' (0-1), 'depth_contrast', 'depth_variance', 'size_ratio'
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = depth_map.shape[:2]
        img_h, img_w = image_shape[:2]

        # Clamp to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return {"score": 0.0, "depth_contrast": 0.0,
                    "depth_variance": 0.0, "size_ratio": 0.0}

        # --- Signal 1: Depth Contrast ---
        # Compare pothole depth to surrounding road surface
        pothole_region = depth_map[y1:y2, x1:x2]
        pothole_mean = np.mean(pothole_region)

        # Expand bbox by 80% to get road context, excluding the pothole itself
        pad_w = int((x2 - x1) * 0.8)
        pad_h = int((y2 - y1) * 0.8)
        sx1 = max(0, x1 - pad_w)
        sy1 = max(0, y1 - pad_h)
        sx2 = min(w, x2 + pad_w)
        sy2 = min(h, y2 + pad_h)

        # Create mask: surrounding area minus the pothole
        surround_mask = np.zeros((sy2 - sy1, sx2 - sx1), dtype=bool)
        surround_mask[:, :] = True
        # Zero out the pothole region within the surrounding patch
        inner_y1 = y1 - sy1
        inner_y2 = y2 - sy1
        inner_x1 = x1 - sx1
        inner_x2 = x2 - sx1
        surround_mask[inner_y1:inner_y2, inner_x1:inner_x2] = False

        surround_patch = depth_map[sy1:sy2, sx1:sx2]
        road_pixels = surround_patch[surround_mask]

        if len(road_pixels) > 0:
            road_mean = np.mean(road_pixels)
            # Normalize the depth difference by the local depth range
            local_range = max(surround_patch.max() - surround_patch.min(), 1e-6)
            depth_contrast = abs(pothole_mean - road_mean) / local_range
        else:
            depth_contrast = 0.0

        # --- Signal 2: Depth Variance ---
        # Deeper, rougher potholes have more internal depth variation
        pothole_std = np.std(pothole_region)
        # Normalize by comparing to the surrounding road's smoothness
        if len(road_pixels) > 0:
            road_std = np.std(road_pixels)
            # Ratio: how much rougher is the pothole than the road
            depth_variance = pothole_std / max(road_std, 1e-6)
            depth_variance = min(depth_variance, 3.0) / 3.0  # normalize to 0-1
        else:
            depth_variance = min(pothole_std * 10, 1.0)

        # --- Signal 3: Size Factor ---
        # Larger potholes (relative to image) are more severe
        pothole_area = (x2 - x1) * (y2 - y1)
        image_area = img_h * img_w
        size_ratio = pothole_area / image_area
        # Scale: a pothole covering 1% of image = 0.2, 5% = 1.0
        size_score = min(1.0, size_ratio * 20)

        # --- Composite Score ---
        # Weighted combination: size matters most, then depth signals
        composite = (
            0.40 * size_score +
            0.30 * min(1.0, depth_contrast * 5) +  # amplify subtle depth diffs
            0.30 * depth_variance
        )
        composite = min(1.0, max(0.0, composite))

        return {
            "score": composite,
            "depth_contrast": depth_contrast,
            "depth_variance": depth_variance,
            "size_ratio": size_ratio,
        }

    @staticmethod
    def classify_severity(score: float) -> tuple:
        """
        Classify pothole severity from composite score.

        Returns:
            (label, color_bgr) tuple
        """
        if score < 0.30:
            return "Low", (0, 200, 0)       # Green
        elif score < 0.55:
            return "Medium", (255, 140, 0)   # Orange
        else:
            return "High", (220, 0, 0)       # Red

    def analyze_frame(self, frame: np.ndarray, detections) -> list:
        """
        Analyze all detected potholes in a frame.

        Args:
            frame: RGB image (numpy array)
            detections: supervision Detections object with xyxy bounding boxes

        Returns:
            List of dicts: bbox, score, severity, color, depth_contrast,
                          depth_variance, size_ratio
        """
        if detections.xyxy is None or len(detections.xyxy) == 0:
            return []

        depth_map = self.estimate_depth(frame)
        results = []

        for bbox in detections.xyxy:
            metrics = self.compute_severity_score(depth_map, bbox, frame.shape)
            severity, color = self.classify_severity(metrics["score"])
            results.append({
                "bbox": bbox,
                "score": metrics["score"],
                "severity": severity,
                "color": color,
                "depth_contrast": metrics["depth_contrast"],
                "depth_variance": metrics["depth_variance"],
                "size_ratio": metrics["size_ratio"],
            })

        return results
