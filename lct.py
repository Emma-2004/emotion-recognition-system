import cv2
import dlib
import numpy as np
import os
from typing import Tuple, List

class MicroExpressionPipeline:
    def __init__(self, predictor_path: str, fps: int = 30, resolution: Tuple[int, int] = (1280, 720)):
        self.fps = fps
        self.width, self.height = resolution
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.eye_roi_range = (36, 48)
        self.mouth_roi_range = (48, 68)

    def detect_landmarks(self, frame: np.ndarray) -> Tuple[bool, List[Tuple[int, int]]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if len(faces) == 0:
            return False, []
        shape = self.predictor(gray, faces[0])
        landmarks = [(p.x, p.y) for p in shape.parts()]
        return True, landmarks

    def crop_roi(self, frame: np.ndarray, landmarks: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        eye_points = landmarks[self.eye_roi_range[0]:self.eye_roi_range[1]]
        eye_x1 = min(p[0] for p in eye_points)
        eye_y1 = min(p[1] for p in eye_points)
        eye_x2 = max(p[0] for p in eye_points)
        eye_y2 = max(p[1] for p in eye_points)
        eye_roi = frame[eye_y1:eye_y2, eye_x1:eye_x2]

        mouth_points = landmarks[self.mouth_roi_range[0]:self.mouth_roi_range[1]]
        mouth_x1 = min(p[0] for p in mouth_points)
        mouth_y1 = min(p[1] for p in mouth_points)
        mouth_x2 = max(p[0] for p in mouth_points)
        mouth_y2 = max(p[1] for p in mouth_points)
        mouth_roi = frame[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
        return eye_roi, mouth_roi

    def normalize_image(self, roi: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        normalized = np.uint8(((roi / 255.0) ** (1 / gamma)) * 255)
        if len(normalized.shape) == 3:
            gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        else:
            return cv2.equalizeHist(normalized)

    def process_pipeline(self, frame: np.ndarray) -> Tuple[bool, dict]:
        success, landmarks = self.detect_landmarks(frame)
        if not success:
            return False, {"error": "未检测到人脸"}
        eye_roi, mouth_roi = self.crop_roi(frame, landmarks)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        gamma = 0.8 if mean_brightness > 180 else 1.2 if mean_brightness < 80 else 1.0
        eye_norm = self.normalize_image(eye_roi, gamma)
        mouth_norm = self.normalize_image(mouth_roi, gamma)
        return True, {
            "eye_roi": eye_roi, "mouth_roi": mouth_roi,
            "eye_normalized": eye_norm, "mouth_normalized": mouth_norm
        }