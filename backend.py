import os
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
        model_file_name = 'best_sku110K_object_identification_yolov11n_model.pt'
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


def draw_annotations_with_gaps(image_path, bboxes, gap_boxes, save_path="annotated_image_with_gaps.jpg"):
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

    # Draw predicted bounding boxes in green
    for (x1, y1, x2, y2) in bboxes:
        # Convert coordinates to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color, 2px thickness

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


