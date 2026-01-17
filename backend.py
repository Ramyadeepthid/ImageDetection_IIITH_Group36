import os
#from turtle import st
import streamlit as st
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv
from PIL import Image
import cv2

# load_dotenv()

class ModelConfig:
    def __init__(self):
        self.yolo_model = self.load_model()

    def load_model(self):
        # model_file_name = os.getenv("model_file_name")
        # model_file_name = 'best_sku110K_object_identification_yolov11n_model.pt'
        model_file_name = 'best_sku110k_partial_data_with_labels.pt'
        # Prepend the /content/ path as the model file is located there
        # full_model_path = os.path.join('/content/', model_file_name)
        print(f"Loading model from file: {model_file_name}")
        model = YOLO(model_file_name)
        return model

    def get_model_predictions(self, source):  # Changed from image_path to source
        results = self.yolo_model.predict(source=source, conf=0.25, show=False, show_labels=True, show_conf=False)
        return results

    def annotate_image(self, results, save_path="annotated_image.jpg"):
        results[0].plot(save=True, filename=save_path)
        return save_path


def cluster_rows(bboxes, row_tol=50):
    """
    bboxes: list of [x1, y1, x2, y2]
    row_tol: vertical tolerance for bottom edges to group into shelves
    """
    bottoms = np.array([y2 for (x1, y1, x2, y2) in bboxes])
    order = np.argsort(bottoms)  # sort by bottom edge

    rows = []
    current = [order[0]]

    for idx in order[1:]:
        prev = current[-1]
        if abs(bottoms[idx] - bottoms[prev]) <= row_tol:
            current.append(idx)
        else:
            rows.append(current)
            current = [idx]

    rows.append(current)
    return rows


# def find_gaps_in_row(row_bboxes, width_factor=0.5):
#     """
#     row_bboxes: list of [x1,y1,x2,y2] for a single shelf row
#     width_factor: threshold relative to median width
#     """
#     # sort left-to-right
#     row_bboxes = sorted(row_bboxes, key=lambda b: b[0])
#     widths = [b[2] - b[0] for b in row_bboxes]
#     median_w = np.median(widths)

#     gaps = []
#     for i in range(len(row_bboxes)-1):
#         cur = row_bboxes[i]
#         nxt = row_bboxes[i+1]
#         gap = nxt[0] - cur[2]

#         if gap > median_w * width_factor:
#             # gap detected
#             gap_box = [
#                 cur[2],
#                 min(cur[1], nxt[1]),
#                 nxt[0],
#                 max(cur[3], nxt[3])
#             ]
#             gaps.append(gap_box)

#     return gaps


def draw_annotations_with_gaps(image_path, bboxes, detected_classes, class_names, gap_boxes, save_path="annotated_image_with_gaps.jpg"):
    """
    Draws bounding boxes from predictions in green and gap boxes in red on the image.
    image_path: Path to the original image.
    bboxes: List of bounding boxes from predictions [x1, y1, x2, y2].
    gap_boxes: List of bounding boxes representing gaps [x1, y1, x2, y2].
    save_path: Path to save the annotated image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    
    # Define font settings and colors
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    box_thickness = 3

    # Colors for bounding boxes and text
    box_color = (255, 0, 0)  # Blue
    text_color = (255, 255, 255) # White
    text_background_color = (255, 0, 0) # Blue


    # Draw predicted bounding boxes in green
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        # Convert coordinates to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color, 2px thickness

        # Prepare label text
        class_id = detected_classes[i]
        label = class_names[class_id]

        # Get text size to position the label properly
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        # Calculate text background position
        text_x = x1
        text_y = y1 - 10 # Position above the box

        # Ensure the label does not go out of image bounds at the top
        if text_y < text_height + baseline:
            text_y = y2 + text_height + 10 # Position below the box if not enough space above

        # Draw filled rectangle for text background
        cv2.rectangle(image,
                    (text_x, text_y - text_height - baseline),
                    (text_x + text_width, text_y),
                    text_background_color, -1)

        # Put text label on the image
        cv2.putText(image, label, (text_x, text_y - baseline),
                    font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # Draw gap boxes in red
    for (x1, y1, x2, y2) in gap_boxes:
        # Convert coordinates to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color, 2px thickness

    cv2.imwrite(save_path, image)
    return save_path

# def calculate_iou(box1, box2):
#     """
#     Calculates the Intersection over Union (IoU) of two bounding boxes.

#     Args:
#         box1 (list or np.array): First bounding box in [x1, y1, x2, y2] format.
#         box2 (list or np.array): Second bounding box in [x1, y1, x2, y2] format.

#     Returns:
#         float: The IoU value, or 0.0 if there is no overlap or union area is zero.
#     """

#     # Determine the coordinates of the intersection rectangle
#     x1_box1, y1_box1, x2_box1, y2_box1 = box1
#     x1_box2, y1_box2, x2_box2, y2_box2 = box2

#     x_left = max(x1_box1, x1_box2)
#     y_top = max(y1_box1, y1_box2)
#     x_right = min(x2_box1, x2_box2)
#     y_bottom = min(y2_box1, y2_box2)

#     # Calculate the area of the intersection rectangle
#     if x_right < x_left or y_bottom < y_top:
#         area_intersection = 0.0
#     else:
#         area_intersection = (x_right - x_left) * (y_bottom - y_top)

#     # Calculate the area of each bounding box
#     area_box1 = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
#     area_box2 = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)

#     # Calculate the Union area
#     area_union = float(area_box1 + area_box2 - area_intersection)

#     # Handle division by zero
#     if area_union == 0:
#         return 0.0

#     # Return the IoU
#     iou = area_intersection / area_union
#     return iou

# def categorize_gaps(empty_shelf_gaps, all_bboxes_list, iou_threshold):
#     """
#     Categorizes empty shelf gaps into overlapping and non-overlapping groups
#     based on Intersection over Union (IoU) with detected object bounding boxes.

#     Args:
#         empty_shelf_gaps (list): A list of bounding boxes representing empty shelf gaps.
#         all_bboxes_list (list): A list of bounding boxes representing detected objects.
#         iou_threshold (float): The IoU threshold to determine if a gap overlaps with an object.

#     Returns:
#         tuple: A tuple containing two lists:
#                - overlapping_gaps (list): Gaps that overlap with any object above the threshold.
#                - non_overlapping_gaps (list): Gaps that do not overlap with any object above the threshold.
#     """
#     overlapping_gaps = []
#     non_overlapping_gaps = []

#     for gap_box in empty_shelf_gaps:
#         is_overlapping = False
#         for object_box in all_bboxes_list:
#             iou = calculate_iou(gap_box, object_box)
#             if iou > iou_threshold:
#                 is_overlapping = True
#                 break # Found an overlap for this gap, no need to check other objects

#         if is_overlapping:
#             overlapping_gaps.append(gap_box)
#         else:
#             non_overlapping_gaps.append(gap_box)

#     return overlapping_gaps, non_overlapping_gaps



##########################################################

import numpy as np

def row_vertical_band(row_bboxes):
    """
    Computes the common vertical free band for a row.
    Returns (y_top, y_bottom) or None if no valid band exists.
    """
    y_top = max(b[1] for b in row_bboxes)
    y_bottom = min(b[3] for b in row_bboxes)

    if y_top >= y_bottom:
        return None

    return y_top, y_bottom


def find_gaps_in_row(row_bboxes, width_factor=0.5):
    """
    Finds horizontal gaps in a single row without overlapping objects.

    row_bboxes: list of [x1, y1, x2, y2]
    width_factor: threshold relative to median width
    """
    # sort left-to-right
    row_bboxes = sorted(row_bboxes, key=lambda b: b[0])

    widths = [b[2] - b[0] for b in row_bboxes]
    median_w = np.median(widths)

    band = row_vertical_band(row_bboxes)
    if band is None:
        return []

    band_top, band_bottom = band
    gaps = []

    for i in range(len(row_bboxes) - 1):
        cur = row_bboxes[i]
        nxt = row_bboxes[i + 1]

        gap_w = nxt[0] - cur[2]
        if gap_w <= median_w * width_factor:
            continue

        gap_box = [
            cur[2],        # x1
            band_top,      # y1
            nxt[0],        # x2
            band_bottom    # y2
        ]

        # final sanity check
        if gap_box[0] < gap_box[2] and gap_box[1] < gap_box[3]:
            gaps.append(gap_box)

    return gaps

def overlaps(box1, box2):
    """
    Returns True if two boxes overlap (any intersection).
    """
    return not (
        box1[2] <= box2[0] or
        box1[0] >= box2[2] or
        box1[3] <= box2[1] or
        box1[1] >= box2[3]
    )


def categorize_gaps(empty_shelf_gaps, all_bboxes_list):
    """
    Categorizes gaps into invalid (overlapping) and valid (true empty).
    Any overlap with an object invalidates the gap.
    """
    overlapping_gaps = []
    non_overlapping_gaps = []

    for gap in empty_shelf_gaps:
        if any(overlaps(gap, obj) for obj in all_bboxes_list):
            overlapping_gaps.append(gap)
        else:
            non_overlapping_gaps.append(gap)

    return overlapping_gaps, non_overlapping_gaps


###########################################################
# For Planogram Compliance
# -----------------------------
# Image utilities
# -----------------------------
import cv2
import numpy as np
#import pytesseract
import json
import matplotlib.pyplot as plt
import google.genai as genai
from typing import List, Dict, Any, Tuple
import os

# -----------------------------
# Image utilities
# -----------------------------
def load_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return img
"""
def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 9)
    return th

def extract_text_annotations(img: np.ndarray) -> List[Dict[str, Any]]:
    pre = preprocess_for_ocr(img)
    data = pytesseract.image_to_data(pre, output_type=pytesseract.Output.DICT)
    annotations = []
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        conf = float(data['conf'][i]) if data['conf'][i] != '-1' else 0.0
        if text and conf > 40:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            annotations.append({
                "text": text,
                "bbox": {"x": x, "y": y, "w": w, "h": h},
                "confidence": conf
            })
    return annotations
"""
def extract_box_annotations(img: np.ndarray) -> List[Dict[str, Any]]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masks = []
    lower_red1, upper_red1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 70, 50]), np.array([180, 255, 255])
    masks.append(cv2.inRange(hsv, lower_red1, upper_red1))
    masks.append(cv2.inRange(hsv, lower_red2, upper_red2))
    lower_blue, upper_blue = np.array([100, 70, 50]), np.array([130, 255, 255])
    masks.append(cv2.inRange(hsv, lower_blue, upper_blue))
    lower_green, upper_green = np.array([40, 70, 50]), np.array([80, 255, 255])
    masks.append(cv2.inRange(hsv, lower_green, upper_green))

    combined = np.zeros_like(masks[0])
    for m in masks:
        combined = cv2.bitwise_or(combined, m)

    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 500:
            boxes.append({"label": None, "bbox": {"x": x, "y": y, "w": w, "h": h}})
    return boxes

def associate_text_to_boxes(text_ann: List[Dict[str, Any]], box_ann: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def center(b):
        return (b["x"] + b["w"] / 2, b["y"] + b["h"] / 2)

    for box in box_ann:
        bx, by = center(box["bbox"])
        best = None
        best_dist = float('inf')
        for t in text_ann:
            tx, ty = center(t["bbox"])
            dist = (bx - tx) ** 2 + (by - ty) ** 2
            if dist < best_dist:
                best_dist = dist
                best = t
        if best and best_dist < 20000:
            box["label"] = best["text"]
    return box_ann

# -----------------------------
# Gemini API interaction
# -----------------------------
def build_llm_payload_for_gemini(image_meta: Dict[str, Any],
                                 text_annotations: List[Dict[str, Any]],
                                 box_annotations: List[Dict[str, Any]]) -> str:
    prompt_text = (
        "You are a retail merchandising expert. Analyze the provided image and its annotations. "
        "Provide me with a summarised report on planogram violation. "
        "This will be important for my vendor contract / compliance. "
        "Provide actionable planogram recommendations. "
        "Output JSON with keys: 'summary', 'recommendations', 'planogram'.\n\n"
        f"Image Metadata: {json.dumps(image_meta)}\n"
        f"Text Annotations: {json.dumps(text_annotations)}\n"
        f"Box Annotations: {json.dumps(box_annotations)}\n"
    )
    return prompt_text

# -----------------------------
# Planogram generation
# -----------------------------
def generate_planogram_from_llm(llm_output: Dict[str, Any]) -> Dict[str, Any]:
    # Extract the 'planogram' part from the LLM's output
    llm_generated_planogram_content = llm_output.get("planogram", {})

    # Initialize the planogram structure for visualization
    visual_planogram = {}

    # Check if the LLM's generated planogram content contains a 'shelves' key
    # with a list of shelves (which it currently doesn't based on the user's output)
    if "shelves" in llm_generated_planogram_content and isinstance(llm_generated_planogram_content["shelves"], list):
        visual_planogram["shelves"] = llm_generated_planogram_content["shelves"]
        visual_planogram["notes"] = llm_generated_planogram_content.get("notes", "Planogram generated by LLM.")
    else:
        # Fallback: create a generic planogram structure for visualization
        # The visualize_planogram expects a list of shelves, each with bays and products
        visual_planogram["shelves"] = [
            {
                "shelf_id": 1,
                "bays": [
                    {"bay_id": 1, "products": [{"sku": "Product A", "facings": 3, "position": [0, 0]}]},
                    {"bay_id": 2, "products": [{"sku": "Product B", "facings": 2, "position": [0, 0]}]}
                ]
            },
            {
                "shelf_id": 2,
                "bays": [
                    {"bay_id": 1, "products": [{"sku": "Product C", "facings": 1, "position": [0, 0]}]},
                    {"bay_id": 2, "products": [{"sku": "Product D", "facings": 4, "position": [0, 0]}]}
                ]
            }
        ]
        visual_planogram["notes"] = "Fallback visual planogram generated as LLM output did not provide a direct 'shelves' structure for visualization."

    # Optionally, you can also store the LLM's original analytical output for reference
    visual_planogram["llm_analysis_summary"] = llm_generated_planogram_content.get("current_state_analysis", "No LLM analysis provided.")
    visual_planogram["llm_recommendations"] = llm_generated_planogram_content.get("proposed_enhancements", [])

    return visual_planogram

# -----------------------------
# Planogram Visualization
# -----------------------------
def visualize_planogram(planogram: Dict[str, Any], figsize: Tuple[int, int] = (10, 6)) -> None:
    shelves = planogram.get("shelves", [])
    fig, ax = plt.subplots(figsize=figsize)
    y_offset = 0
    for shelf in shelves:
        bays = shelf.get("bays", [])
        ax.plot([0, 10], [y_offset, y_offset], color='black', linewidth=2)
        x_offset = 0
        for bay in bays:
            products = bay.get("products", [])
            ax.add_patch(plt.Rectangle((x_offset, y_offset), 2, 1, fill=False, edgecolor='gray'))
            px = x_offset + 0.1
            py = y_offset + 0.1
            for p in products:
                sku = p.get("sku", "SKU")
                facings = p.get("facings", 1)
                for f in range(facings):
                    ax.add_patch(plt.Rectangle((px + f * 0.15, py), 0.12, 0.3, color='skyblue', alpha=0.7))
                ax.text(px, py + 0.35, f"{sku} ({facings}x)", fontsize=8)
            x_offset += 2.2
        y_offset += 1.5

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, y_offset + 0.5)
    ax.set_title("Planogram Visualization")
    ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)

###########################################################
# For  Smart Shelf Dashboard
# -----------------------------

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_dummy_data():
    os.makedirs('data', exist_ok=True)
    skus = [f"SKU_{i:03d}" for i in range(1, 21)]
    categories = ['Beverages', 'Snacks', 'Dairy', 'Frozen', 'Produce']
    
    # 1. Product Velocity & Restock Efficiency Data
    velocity_rows = []
    for _ in range(200):
        sku = np.random.choice(skus)
        alert_time = datetime.now() - timedelta(days=np.random.randint(0, 7), hours=np.random.randint(0, 24))
        # Wait time between 5 mins and 120 mins
        wait_time = np.random.randint(5, 120)
        refill_time = alert_time + timedelta(minutes=wait_time)
        velocity_rows.append({
            'sku': sku,
            'category': np.random.choice(categories),
            'alert_timestamp': alert_time,
            'refill_timestamp': refill_time,
            'wait_time_mins': wait_time,
            'units_refilled': np.random.randint(10, 50)
        })
    pd.DataFrame(velocity_rows).to_csv('data/product_velocity.csv', index=False)

    # 2. Lost Revenue Data
    revenue_rows = []
    for sku in skus:
        oos_duration = np.random.uniform(1.0, 8.0) # hours
        avg_velocity = np.random.uniform(5.0, 15.0) # units/hour
        unit_price = np.random.uniform(2.5, 25.0)
        revenue_rows.append({
            'sku': sku,
            'oos_duration_hours': round(oos_duration, 2),
            'potential_revenue_lost': round(oos_duration * avg_velocity * unit_price, 2)
        })
    pd.DataFrame(revenue_rows).to_csv('data/lost_revenue.csv', index=False)