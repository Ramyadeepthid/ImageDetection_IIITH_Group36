
# import os
# import numpy as np
# from ultralytics import YOLO
# from dotenv import load_dotenv
# from PIL import Image
# import cv2

# load_dotenv()

# class ModelConfig:
#     def __init__(self):
#         self.yolo_model = self.load_model()

#     def load_model(self):
#         model_file_name = os.getenv("model_file_name")
#         # Prepend the /content/ path as the model file is located there
#         # full_model_path = os.path.join('/content/', model_file_name)
#         print(f"Loading model from file: {model_file_name}")
#         model = YOLO(model_file_name)
#         return model

#     def get_model_predictions(self, image_path):
#         results = self.yolo_model.predict(source=image_path, conf=0.25, show=False, show_labels=True, show_conf=False)
#         return results

#     def annotate_image(self, results, save_path="annotated_image.jpg"):
#         results[0].plot(save=True, filename=save_path)
#         return save_path

# def cluster_rows(bboxes, row_tol=50):
#     """
#     bboxes: list of [x1, y1, x2, y2]
#     row_tol: vertical tolerance for bottom edges to group into shelves
#     """
#     bottoms = np.array([y2 for (x1, y1, x2, y2) in bboxes])
#     order = np.argsort(bottoms)  # sort by bottom edge

#     rows = []
#     current = [order[0]]

#     for idx in order[1:]:
#         prev = current[-1]
#         if abs(bottoms[idx] - bottoms[prev]) <= row_tol:
#             current.append(idx)
#         else:
#             rows.append(current)
#             current = [idx]

#     rows.append(current)
#     return rows


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


import os
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv
from PIL import Image
import cv2

load_dotenv()

class ModelConfig:
    def __init__(self):
        self.yolo_model = self.load_model()

    def load_model(self):
        model_file_name = os.getenv("model_file_name")
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


def find_gaps_in_row(row_bboxes, width_factor=0.5):
    """
    row_bboxes: list of [x1,y1,x2,y2] for a single shelf row
    width_factor: threshold relative to median width
    """
    # sort left-to-right
    row_bboxes = sorted(row_bboxes, key=lambda b: b[0])
    widths = [b[2] - b[0] for b in row_bboxes]
    median_w = np.median(widths)

    gaps = []
    for i in range(len(row_bboxes)-1):
        cur = row_bboxes[i]
        nxt = row_bboxes[i+1]
        gap = nxt[0] - cur[2]

        if gap > median_w * width_factor:
            # gap detected
            gap_box = [
                cur[2],
                min(cur[1], nxt[1]),
                nxt[0],
                max(cur[3], nxt[3])
            ]
            gaps.append(gap_box)

    return gaps


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
