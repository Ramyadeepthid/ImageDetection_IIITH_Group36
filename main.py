try:
    
# import streamlit as st
# from PIL import Image
# from io import BytesIO
# import base64
# import random
# import textwrap
# from backend import ModelConfig, cluster_rows, find_gaps_in_row

# st.set_page_config(layout="wide", page_title="Group 36 Object Detection Image Viewer")

# try:
#     print("Loading YOLO model...")
#     # Initialize ModelConfig which loads the YOLO model
#     model_config = ModelConfig()
#     model = model_config.yolo_model # Access the loaded YOLO model
#     print("YOLO model loaded successfully.")
# except Exception as e:
#     print(f"Error loading YOLO model: {e}")
#     st.error(f"Error loading YOLO model: {e}. Make sure the model file is present and .env is configured.")
#     st.stop() # Stop execution if model can't be loaded
# # -------------------------
# # Replace this with detection/classification function.
# # It must accept image bytes (or a PIL image) and return a class label string.
# # -------------------------

# def predict_class_for_image(pil_image: Image.Image) -> str:
#     """
#     Placeholder classifier - replace with your model inference.

#     Example replacement:
#         img_bytes = pil_image_to_bytes(pil_image)
#         result = your_model.predict(img_bytes)
#         return result['class_name']

#     For demo we randomly assign classes.
#     """
#     model
#     classes = ["class1", "class2", "class3"]
#     return random.choice(classes)

# # Utility: convert PIL image to base64 data URL (for inline HTML img)
# def pil_image_to_data_url(pil_img: Image.Image, fmt="JPEG", max_size=None):
#     img = pil_img.copy()
#     if max_size:
#         img.thumbnail(max_size, Image.Resampling.LANCZOS)
#     buf = BytesIO()
#     img.save(buf, format=fmt, quality=85)
#     byte_im = buf.getvalue()
#     b64 = base64.b64encode(byte_im).decode("utf-8")
#     return f"data:image/{fmt.lower()};base64,{b64}"


# # Utility: safe filename/key for query param (we'll use index)
# def make_key(idx: int, filename: str):
#     return f"{idx}"

# st.title("Yolo Object detection Image Viewer — upload, detect, and group")

# import streamlit as st

# MAX_FILES = 20
# MAX_SIZE_PER_FILE_MB = 5   # per image
# MAX_TOTAL_SIZE_MB = 50     # all images combined

# def check_image_limits(uploaded_files):
#     if not uploaded_files:
#         return True, None

#     # 1. Check number of files
#     if len(uploaded_files) > MAX_FILES:
#         return False, f"Too many files! Maximum allowed: {MAX_FILES}"

#     total_size = 0
#     for file in uploaded_files:
#         # 2. Check file type (optional but recommended)
#         if not file.type.startswith('image/'):
#             return False, f"File '{file.name}' is not an image!"

#         # 3. Check individual file size
#         file_size_mb = len(file.getvalue()) / (1024 * 1024)
#         if file_size_mb > MAX_SIZE_PER_FILE_MB:
#             return False, f"File '{file.name}' is too large! Max {MAX_SIZE_PER_FILE_MB} MB per image."

#         total_size += file_size_mb

#     # 4. Check total size
#     if total_size > MAX_TOTAL_SIZE_MB:
#         return False, f"Total size exceeds {MAX_TOTAL_SIZE_MB} MB! Current: {total_size:.1f} MB"

#     return True, None

# uploaded_files = st.file_uploader(
#     "Choose images",
#     type=["png", "jpg", "jpeg", "webp", "bmp"],
#     accept_multiple_files=True,
#     help=f"Max {MAX_FILES} images | {MAX_SIZE_PER_FILE_MB} MB per image | {MAX_TOTAL_SIZE_MB} MB total",
#     label_visibility="collapsed"
# )

# if uploaded_files:
#     is_valid, error_msg = check_image_limits(uploaded_files)
    
#     if not is_valid:
#         st.error(error_msg)
#         uploaded_files = None  # ignore the files if validation fails
#     else:
#         st.success(f"{len(uploaded_files)} image(s) uploaded successfully!")

# # Button to run detection, or auto-run once files are uploaded
# run_detection = st.button("Run detection and group")  # user explicit click

# if uploaded_files and (run_detection or st.session_state.get("auto_run_once") is None):
#     # mark that we ran once; prevents auto-re-running every interaction unless user clicks again
#     st.session_state["auto_run_once"] = True

#     # Read, predict, and group
#     groups = {}  # class_name -> list of dicts: {key, filename, pil, thumb_data_url, full_bytes}
#     for idx, uploaded in enumerate(uploaded_files):
#         raw = uploaded.read()
#         try:
#             pil = Image.open(BytesIO(raw)).convert("RGB")
#         except Exception as e:
#             st.warning(f"Couldn't open {uploaded.name}: {e}")
#             continue

#         # run prediction/classification (replace with your model)
#         cls = predict_class_for_image(pil)

#         key = make_key(idx, uploaded.name)
#         thumb_data_url = pil_image_to_data_url(pil, fmt="JPEG", max_size=(500,500))
#         # store full bytes for download/view
#         entry = {
#             "key": key,
#             "filename": uploaded.name,
#             "pil": pil,
#             "thumb_url": thumb_data_url,
#             "full_bytes": raw
#         }
#         groups.setdefault(cls, []).append(entry)

#     # persist groups in session state for later rendering/clicking
#     st.session_state["groups"] = groups

# # Load groups from session state if present
# groups = st.session_state.get("groups", {})

# if not groups:
#     st.info("Upload images and click **Run detection and group** to see grouped thumbnails.")
#     st.stop()

# # ---- READ QUERY PARAMS ----
# qp = st.query_params
# selected_key = None
# if "selected" in qp and len(qp["selected"]) > 0:
#     selected_key = qp["selected"][0]


# # Helper to generate the HTML block for a class container with horizontal scrolling
# def class_container_html(class_name: str, entries: list):
#     header = f"<div style='font-weight:600; padding:8px 0;'>{class_name} ({len(entries)})</div>"

#     container_style = (
#         "border:2px solid #444;"
#         "border-radius:18px;"
#         "padding:12px;"
#         "margin-bottom:24px;"
#         "overflow-x:auto;"
#         "white-space:nowrap;"
#     )
#     container_div_open = f"<div style='{container_style}'>"
#     container_div_close = "</div>"

#     thumbs_html = []
#     for e in entries:
#         card_style = "display:inline-block;width:320px;margin-right:12px;text-align:center;cursor:pointer;"
#         img_style = "display:block;height:240px;width:auto;margin-bottom:6px;border:1px solid #ddd;border-radius:6px;"
#         fn = e['filename']
#         key = e['key']

#         thumb_tag = (
#             f"<div style='{card_style}' onclick=\"window.parent.streamlit.setComponentValue({{key: '{key}'}})\">"
#             f"    <img src='{e['thumb_url']}' style='{img_style}' />"
#             f"  <div style='font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>{fn}</div>"
#             f"</div>"
#         )
#         thumbs_html.append(thumb_tag)

#     return header + container_div_open + "".join(thumbs_html) + container_div_close


# # Render each class container with scrollable thumbnails and dropdown selector
# st.write("### Grouped images")
# for class_name, entries in groups.items():
#     st.markdown(f"**{class_name} ({len(entries)})**")
    
#     # Create horizontal scrollable container with interactive thumbnails
#     html = class_container_html(class_name, entries)
#     st.markdown(html, unsafe_allow_html=True)
    
#     # Create dropdown and action buttons
#     col1, col2, col3 = st.columns([2, 1, 1])
    
#     # Dropdown to select image
#     image_options = {e["filename"]: e["key"] for e in entries}
#     with col1:
#         selected_filename = st.selectbox(
#             "Select image",
#             options=list(image_options.keys()),
#             key=f"select_{class_name}",
#             label_visibility="collapsed"
#         )
#         selected_key_from_dropdown = image_options[selected_filename]
    
#     # Preview button
#     with col2:
#         if st.button("👁️ Preview", key=f"preview_{class_name}", use_container_width=True):
#             st.session_state["selected_key"] = selected_key_from_dropdown
#             st.rerun()
    
#     # Download button
#     with col3:
#         # Find the entry for this key to get the full bytes
#         for e in entries:
#             if e["key"] == selected_key_from_dropdown:
#                 st.download_button(
#                     "📥 Download",
#                     data=e["full_bytes"],
#                     file_name=e["filename"],
#                     key=f"download_{class_name}",
#                     use_container_width=True
#                 )
#                 break
    
#     st.divider()

# # If a selection is made, show full image in expander
# selected_key = st.session_state.get("selected_key")
# full_entry = None

# if selected_key:
#     for cls_entries in groups.values():
#         for e in cls_entries:
#             if e["key"] == selected_key:
#                 full_entry = e
#                 break
#         if full_entry:
#             break

# if full_entry:
#     with st.expander(f"📸 **{full_entry['filename']}** (Click to expand/collapse)", expanded=True):
#         st.image(full_entry["pil"], use_container_width=True)
#         col1, col2 = st.columns(2)
#         with col1:
#             st.download_button(
#                 "📥 Download original",
#                 data=full_entry["full_bytes"],
#                 file_name=full_entry["filename"],
#                 use_container_width=True
#             )
#         with col2:
#             if st.button("✕ Close", use_container_width=True):
#                 st.session_state["selected_key"] = None
#                 st.rerun()
# else:
#     st.info("👆 Select an image from the dropdown and click **Preview** to view it below.")

# # Footer: small hint about replacing the detection function
# st.markdown(
#     """
#     <hr>
#     <small>Replace `predict_class_for_image()` with your object detection / classification model's inference
#     and return the appropriate class label for grouping.</small>
#     """,
#     unsafe_allow_html=True
# )
    pass
except Exception as ex:
    pass

import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import random
import textwrap

# Import from backend
from backend import ModelConfig, find_gaps_in_row, categorize_gaps, cluster_rows ,draw_annotations_with_gaps

st.set_page_config(layout="wide", page_title="Group 36 Object Detection Image Viewer")
page = st.sidebar.radio(
    "Go to",
    ["Object Detection", "Planogram Compliance"]
)
if page == "Home":
    st.title("Home")
    st.write("Welcome.")
elif page == "Object Detection":
    # Hide Streamlit's default file uploader preview (from previous response)
    hide_streamlit_style = """
        <style>
            /* Hide the default preview of uploaded files (thumbnails + names) */
            section[data-testid="stFileUploader"] div:has(> div > img) {
                display: none !important;
            }
            section[data-testid="stFileUploader"] div[data-testid="stMarkdownContainer"] {
                display: none !important;
            }
            .uploadedFile { display: none !important; }
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # Load model once (assuming .env is loaded in backend)
    model_config = ModelConfig()

    # -------------------------
    # Detection/classification function using backend logic
    # -------------------------
    def predict_class_for_image(pil_image: Image.Image): # Now returns tuple (str, list, list)
        """
        Run YOLO, cluster rows, find gaps, and classify based on total gaps.
        """
        # Get YOLO predictions (passes PIL directly)
        results = model_config.get_model_predictions(pil_image)
        
        # Extract bounding boxes (xyxy format as list of lists)
        if not results[0].boxes:  # No detections
            # Return default class and empty lists for bboxes and gaps
            return "Class1", [], []
        
        bboxes = results[0].boxes.xyxy.cpu().numpy().tolist()  # [[x1,y1,x2,y2], ...]
        
        # Cluster into rows
        rows = cluster_rows(bboxes)
        
        # Find gaps across all rows
        all_gaps = []
        for row_indices in rows:
            row_bboxes = [bboxes[i] for i in row_indices]
            gaps = find_gaps_in_row(row_bboxes)
            overlapping_gaps, non_overlapping_gaps = categorize_gaps(gaps, bboxes)
            all_gaps.extend(non_overlapping_gaps)
        


        # Classify based on total gaps
        total_gaps = len(all_gaps)
        if total_gaps == 0 or total_gaps == 1:  # Treating 1 as "minimal/no gaps"—adjust if needed
            cls = "Class1"
        elif 2 <= total_gaps <= 4:
            cls = "Class2"
        else:  # >4
            cls = "Class3"
            
        return cls, bboxes, all_gaps

    # Utility: convert PIL image to base64 data URL (for inline HTML img)
    def pil_image_to_data_url(pil_img: Image.Image, fmt="JPEG", max_size=None):
        img = pil_img.copy()
        if max_size:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format=fmt, quality=85)
        byte_im = buf.getvalue()
        b64 = base64.b64encode(byte_im).decode("utf-8")
        return f"data:image/{fmt.lower()};base64,{b64}"

    # Utility: safe filename/key for query param (we'll use index)
    def make_key(idx: int, filename: str):
        return f"{idx}"

    st.title("Yolo Object detection Image Viewer — upload, detect, and group")

    MAX_FILES = 20
    MAX_SIZE_PER_FILE_MB = 5   # per image
    MAX_TOTAL_SIZE_MB = 50     # all images combined

    def check_image_limits(uploaded_files):
        if not uploaded_files:
            return True, None

        # 1. Check number of files
        if len(uploaded_files) > MAX_FILES:
            return False, f"Too many files! Maximum allowed: {MAX_FILES}"

        total_size = 0
        for file in uploaded_files:
            # 2. Check file type (optional but recommended)
            if not file.type.startswith('image/'):
                return False, f"File '{file.name}' is not an image!"

            # 3. Check individual file size
            file_size_mb = len(file.getvalue()) / (1024 * 1024)
            if file_size_mb > MAX_SIZE_PER_FILE_MB:
                return False, f"File '{file.name}' is too large! Max {MAX_SIZE_PER_FILE_MB} MB per image."

            total_size += file_size_mb

        # 4. Check total size
        if total_size > MAX_TOTAL_SIZE_MB:
            return False, f"Total size exceeds {MAX_TOTAL_SIZE_MB} MB! Current: {total_size:.1f} MB"

        return True, None

    uploaded_files = st.file_uploader(
        "Choose images",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        accept_multiple_files=True,
        help=f"Max {MAX_FILES} images | {MAX_SIZE_PER_FILE_MB} MB per image | {MAX_TOTAL_SIZE_MB} MB total"
    )

    if uploaded_files:
        is_valid, error_msg = check_image_limits(uploaded_files)
        
        if not is_valid:
            st.error(error_msg)
            uploaded_files = None  # ignore the files if validation fails
        else:
            st.success(f"{len(uploaded_files)} image(s) uploaded successfully!")

    # Button to run detection, or auto-run once files are uploaded
    run_detection = st.button("Run detection and group")  # user explicit click

    if uploaded_files and (run_detection or st.session_state.get("auto_run_once") is None):
        # mark that we ran once; prevents auto-re-running every interaction unless user clicks again
        st.session_state["auto_run_once"] = True

        # Read, predict, and group
        groups = {}  # class_name -> list of dicts: {key, filename, pil, thumb_data_url, full_bytes}
        for idx, uploaded in enumerate(uploaded_files):
            raw = uploaded.read()
            try:
                pil = Image.open(BytesIO(raw)).convert("RGB")
            except Exception as e:
                st.warning(f"Couldn't open {uploaded.name}: {e}")
                continue

            # run prediction/classification (now using backend logic)
            cls, bboxes, all_gaps = predict_class_for_image(pil)

            key = make_key(idx, uploaded.name)
            
            import os
            # Create a temporary file to save the original image for OpenCV to read
            temp_original_image_path = f"original_{idx}.jpg"
            pil.save(temp_original_image_path)

            # Draw annotations and gaps
            annotated_path = draw_annotations_with_gaps(temp_original_image_path, bboxes, all_gaps, save_path=f"annotated_with_gaps_{idx}.jpg")
            annotated_pil = Image.open(annotated_path)
            thumb_data_url = pil_image_to_data_url(annotated_pil, fmt="JPEG", max_size=(500,500))
            pil = annotated_pil  # Use for full preview too

            # Clean up temporary files
            os.remove(temp_original_image_path)
            os.remove(annotated_path)

            # store full bytes for download/view
            entry = {
                "key": key,
                "filename": uploaded.name,
                "pil": pil,
                "thumb_url": thumb_data_url,
                "full_bytes": raw
            }
            groups.setdefault(cls, []).append(entry)

        # persist groups in session state for later rendering/clicking
        st.session_state["groups"] = groups

    # Load groups from session state if present
    groups = st.session_state.get("groups", {})
    if not groups:
        st.info("Upload images and click **Run detection and group** to see grouped thumbnails.")
        st.stop()

    # ---- READ QUERY PARAMS ----
    qp = st.query_params
    selected_key = None
    if "selected" in qp and len(qp["selected"]) > 0:
        selected_key = qp["selected"][0]

    # Helper to generate the HTML block for a class container with horizontal scrolling
    def class_container_html(class_name: str, entries: list):
        header = f"<div style='font-weight:600; padding:8px 0;'>{class_name} ({len(entries)})</div>"

        container_style = (
            "border:2px solid #444;"
            "border-radius:18px;"
            "padding:12px;"
            "margin-bottom:24px;"
            "overflow-x:auto;"
            "white-space:nowrap;"
        )
        container_div_open = f"<div style='{container_style}'>"
        container_div_close = "</div>"

        thumbs_html = []
        for e in entries:
            card_style = "display:inline-block;width:320px;margin-right:12px;text-align:center;cursor:pointer;"
            img_style = "display:block;height:240px;width:auto;margin-bottom:6px;border:1px solid #ddd;border-radius:6px;"
            fn = e['filename']
            key = e['key']

            thumb_tag = (
                f"<div style='{card_style}' onclick=\"window.parent.streamlit.setComponentValue({{key: '{key}'}})\">"
                f"    <img src='{e['thumb_url']}' style='{img_style}' />"
                f"  <div style='font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>{fn}</div>"
                f"</div>"
            )
            thumbs_html.append(thumb_tag)

        return header + container_div_open + "".join(thumbs_html) + container_div_close

    # Render each class container with scrollable thumbnails and dropdown selector
    st.write("### Grouped images")
    for class_name, entries in groups.items():
        st.markdown(f"**{class_name} ({len(entries)})**")
        
        # Create horizontal scrollable container with interactive thumbnails
        html = class_container_html(class_name, entries)
        st.markdown(html, unsafe_allow_html=True)
        
        # Create dropdown and action buttons
        col1, col2, col3 = st.columns([2, 1, 1])
        
        # Dropdown to select image
        image_options = {e["filename"]: e["key"] for e in entries}
        with col1:
            selected_filename = st.selectbox(
                "Select image",
                options=list(image_options.keys()),
                key=f"select_{class_name}",
                label_visibility="collapsed"
            )
            selected_key_from_dropdown = image_options[selected_filename]
        
        # Preview button
        with col2:
            if st.button("👁️ Preview", key=f"preview_{class_name}", width='stretch'):
                st.session_state["selected_key"] = selected_key_from_dropdown
                st.rerun()
        
        # Download button
        with col3:
            # Find the entry for this key to get the full bytes
            for e in entries:
                if e["key"] == selected_key_from_dropdown:
                    st.download_button(
                        "📥 Download",
                        data=e["full_bytes"],
                        file_name=e["filename"],
                        key=f"download_{class_name}",
                        width='stretch'
                    )
                    break
        
        st.divider()

    # If a selection is made, show full image in expander
    selected_key = st.session_state.get("selected_key")
    full_entry = None

    if selected_key:
        for cls_entries in groups.values():
            for e in cls_entries:
                if e["key"] == selected_key:
                    full_entry = e
                    break
            if full_entry:
                break

    if full_entry:
        with st.expander(f"📸 **{full_entry['filename']}** (Click to expand/collapse)", expanded=True):
            st.image(full_entry["pil"], width='stretch')
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download original",
                    data=full_entry["full_bytes"],
                    file_name=full_entry["filename"],
                    width='stretch'
                )
            with col2:
                if st.button("✕ Close", width='stretch'):
                    st.session_state["selected_key"] = None
                    st.rerun()
    else:
        st.info("👆 Select an image from the dropdown and click **Preview** to view it below.")

    # Footer: small hint about replacing the detection function
    st.markdown(
        """
        <hr>
        <small>YOLO detection and gap-based classification integrated from backend.py.</small>
        """,
        unsafe_allow_html=True
    )
elif page == "Planogram Compliance":
    st.title("Planogram Compliance")
    st.write("Planogram Compliance logic to be added here.")
