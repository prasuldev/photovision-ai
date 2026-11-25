import logging

from PIL import Image, UnidentifiedImageError
import streamlit as st

from inference import read_photo
from photo_analysis import (
    DETECTION_THRESHOLD,
    annotate_objects,
    describe_photo,
    find_objects,
    identify_subject,
    load_captioner,
    load_classifier,
    load_detector,
)

DESCRIBE = "Describe the photo"
DETECT = "Find objects"
CLASSIFY = "Identify the main subject"
ANALYSIS_MODES = {
    DESCRIBE: (load_captioner, describe_photo),
    DETECT: (load_detector, find_objects),
    CLASSIFY: (load_classifier, identify_subject),
}


@st.cache_resource(max_entries=3)
def get_model(mode):
    loader, _ = ANALYSIS_MODES[mode]
    return loader()


def clear_result():
    st.session_state.pop("result", None)


st.set_page_config(page_title="PhotoVision AI", page_icon="📷")
st.title("PhotoVision AI")
st.write("Upload a photo to describe the scene, find objects, or identify the main subject.")
uploaded = st.file_uploader(
    "Upload a photo",
    type=["jpg", "jpeg", "png", "webp"],
    help="JPEG, PNG or WebP. Up to 20 MB.",
    on_change=clear_result,
)

if uploaded is None:
    st.stop()

try:
    photo = read_photo(uploaded)
except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
    st.error("This photo could not be opened. Try a valid JPEG, PNG or WebP file.")
    st.stop()

photo.thumbnail((1600, 1600))
mode = st.segmented_control(
    "Analysis mode",
    list(ANALYSIS_MODES),
    default=DESCRIBE,
    required=True,
    on_change=clear_result,
)

if st.button("Analyze photo", type="primary"):
    clear_result()
    with st.spinner("Analyzing photo. The first run may download model weights…"):
        try:
            _, analyze = ANALYSIS_MODES[mode]
            st.session_state.result = analyze(get_model(mode), photo)
        except Exception:
            logging.getLogger(__name__).exception("Photo analysis failed in %s", mode)
            st.error("Analysis could not run. Check your internet connection for the first model download and try again.")

result = st.session_state.get("result")
if mode == DETECT and result:
    st.image(annotate_objects(photo, result), caption="Detected objects", width="stretch")
else:
    st.image(photo, caption="Uploaded photo", width="stretch")

if result is not None:
    if mode == DESCRIBE:
        st.subheader("Description")
        st.text(result)
    elif mode == DETECT:
        if result:
            st.subheader("Detected objects")
            st.table([
                {"Object": label.capitalize(), "Model confidence": f"{score:.1%}"}
                for _, label, score in result
            ])
        else:
            st.info(f"No supported objects were detected above {DETECTION_THRESHOLD:.0%} confidence.")
    else:
        label, score = result
        st.subheader(label.capitalize())
        st.metric("Model confidence", f"{score:.1%}")
