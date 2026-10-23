"""Métricas de máscara para comparar el muro exterior predicho con el anotado."""

from __future__ import annotations

import cv2
import numpy as np


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    union = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / union) if union else 1.0


def tolerant_prf(pred: np.ndarray, gt: np.ndarray, tolerance_px: int) -> tuple[float, float, float]:
    """Precisión/recall con holgura: un píxel acierta si cae a <= tolerance_px del otro.

    La anotación y el dibujo no coinciden al píxel (cara exterior vs eje del muro), así que
    el IoU estricto castiga diferencias que no importan.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance_px + 1,) * 2)
    gt_near = cv2.dilate(gt.astype(np.uint8), kernel) > 0
    pred_near = cv2.dilate(pred.astype(np.uint8), kernel) > 0
    n_pred, n_gt = pred.sum(), gt.sum()
    precision = float((pred & gt_near).sum() / n_pred) if n_pred else 0.0
    recall = float((gt & pred_near).sum() / n_gt) if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def comparison_image(gray: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Verde = solo anotado (faltó), rosado = solo predicho (sobra), azul = ambos."""
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color = bgr.copy()
    color[gt & ~pred] = (60, 200, 60)
    color[pred & ~gt] = (160, 64, 255)
    color[pred & gt] = (220, 120, 30)
    return cv2.addWeighted(color, 0.7, bgr, 0.3, 0)
