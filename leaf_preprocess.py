# ====================================================
# leaf_preprocess.py - Leaf Extraction for Training/Inference
# ====================================================
import cv2
import numpy as np
from PIL import Image

def preprocess_leaf(image: Image.Image, min_area: int = 500) -> Image.Image:
    """
    Preprocess input so model receives only the leaf (masked + cropped).
    - Detects largest green contour as leaf.
    - Keeps inner dark/diseased spots (no hole-filling).
    - If detection fails, returns original image.

    Args:
        image (PIL.Image): Input image
        min_area (int): Minimum area to consider as valid leaf

    Returns:
        PIL.Image: Processed leaf or original image
    """
    # Convert PIL -> OpenCV (BGR)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert to HSV
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

    # Green detection range (tweak if dataset lighting varies)
    lower = np.array([25, 40, 40])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Smooth mask borders but DO NOT remove holes inside leaf
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return image  # fail → return original

    # Largest contour = leaf
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return image  # too small → return original

    # Create mask for just the leaf shape (keeps interior as-is)
    leaf_mask = np.zeros_like(mask)
    cv2.drawContours(leaf_mask, [largest], -1, 255, thickness=-1)

    # Apply mask (keeps inner details, disease spots intact)
    leaf_only = cv2.bitwise_and(img_cv, img_cv, mask=leaf_mask)

    # Crop to bounding box of leaf
    x, y, w, h = cv2.boundingRect(largest)
    leaf_cropped = leaf_only[y:y+h, x:x+w]

    # Convert back to PIL
    return Image.fromarray(cv2.cvtColor(leaf_cropped, cv2.COLOR_BGR2RGB))
