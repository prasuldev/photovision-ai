import os
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import streamlit as st

from inference import classify, load_model, read_photo


@st.cache_resource(max_entries=2)
def cached_model(path: Path, modified_ns: int, size: int):
    # File metadata makes replacing the checkpoint invalidate the cache.
    return load_model(path)


st.set_page_config(page_title="LionVision AI", page_icon="🦁")
st.title("LionVision AI")
st.write("Upload a wildlife photo to see whether the model classifies it as a lion.")
uploaded = st.file_uploader(
    "Upload a photo",
    type=["jpg", "jpeg", "png", "webp"],
    help="JPEG, PNG or WebP. Up to 20 MB.",
)

default_checkpoint = Path(__file__).with_name("lion_classifier.pth")
checkpoint = Path(os.environ.get("LIONVISION_MODEL_PATH", default_checkpoint)).expanduser()
try:
    checkpoint_info = checkpoint.stat() if checkpoint.is_file() else None
except OSError:
    checkpoint_info = None

if checkpoint_info is None:
    st.info("Model weights are missing. You can preview photos while inference is unavailable.")

if uploaded is not None:
    try:
        photo = read_photo(uploaded)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        st.error("This photo could not be opened. Try a valid JPEG, PNG or WebP file.")
        st.stop()

    st.image(photo, width="stretch")
    if checkpoint_info is not None and st.button("Classify photo", type="primary"):
        with st.spinner("Classifying photo…"):
            try:
                model = cached_model(checkpoint, checkpoint_info.st_mtime_ns, checkpoint_info.st_size)
                label, confidence = classify(model, photo)
            except Exception:
                st.error("The classifier could not run. Check that the configured file is a compatible LionVision AI checkpoint.")
                st.stop()
        st.subheader(label)
        st.metric("Model confidence", f"{confidence:.1%}")
        st.caption("This score comes from the model and can be high even when a prediction is wrong.")
