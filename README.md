# LionVision AI

LionVision AI is a Streamlit app for classifying wildlife photos. Drag in a JPEG,
PNG or WebP file (up to 20 MB), preview it, then select **Classify photo**. With
compatible model weights installed, the app returns **Lion** or **Not a lion**
and the winning class's softmax score.

## Run locally

Use Python 3.11 or 3.13. Open the repository folder in VS Code and run these
commands in its terminal (macOS or Linux).

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

Open [LionVision AI](http://localhost:8501).

## Enable predictions

Put your trained `lion_classifier.pth` beside `app.py`, or set
`LIONVISION_MODEL_PATH` to its full path before starting Streamlit. Model weights
and training photos live outside Git. A checkout without a checkpoint supports
uploads and previews and shows that inference is unavailable.

Use a trusted PyTorch ResNet-18 state dictionary produced by the existing
`lionvision.py` training script. The app uses CPU inference and matches training
preprocessing by resizing RGB images to 224 × 224 and converting pixels to
tensors. The original `ImageFolder` class order is `lions`, then `not_lions`.

## Train with your own images

Place labeled photos in the following folders. Keep the validation images
separate from the training images.

```text
data/
  train/
    lions/
    not_lions/
  val/
    lions/
    not_lions/
```

Run `python lionvision.py` from the repository root. The script downloads
ImageNet initialization weights if they are absent, prints validation accuracy,
and saves `lion_classifier.pth`. It then attempts an example prediction using
`data/train/lions/lion1.jpg`. Set `test_image` in the script to an existing
photo if that sample path is absent.

## Model scope

The current checkpoint classifies the whole image. Face localization and
individual lion identification would require suitable labeled data and a
different model. The confidence score is uncalibrated, and the app's smoke
tests do not establish accuracy on new photos.

Created by Prasuldev.
