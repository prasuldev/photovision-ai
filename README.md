# PhotoVision AI

Upload a photo once and choose how to analyze it in Streamlit.

| Mode | Result | Pretrained model |
| --- | --- | --- |
| Describe the photo (default) | A short English description of the scene | [Salesforce BLIP base](https://huggingface.co/Salesforce/blip-image-captioning-base) |
| Find objects | Labeled boxes and scores for supported objects above 45% model confidence | [Torchvision SSDLite trained on COCO](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html) |
| Identify the main subject | The highest-scoring category and its score | [Torchvision ResNet-18 trained on ImageNet](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) |

## Run locally

Open the repository folder in VS Code. Use Python 3.11 or 3.13, then run these
commands in the terminal (macOS or Linux).

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

Open [PhotoVision AI](http://127.0.0.1:8501). Drag in a JPEG, PNG, or WebP
file up to 20 MB, choose a mode, and select **Analyze photo**. You can switch
modes without uploading the image again.

Each model downloads its weights on first use, then reuses the local cache.
The description model is roughly 1 GB, so allow extra time and disk space for
that first download. Inference runs on the server's CPU. With the command
above, the server runs on your computer, and model downloads do not include
your photos. No API key is required.

## Interpreting results

The object detector and subject classifier use the categories their models were
trained on. The same photo can produce a useful caption and no detected boxes.
Captions can omit details or misidentify objects, and model scores are
uncalibrated.

## Original lion classifier

The original `lionvision.py` training script and `inference.py` checkpoint
loader are still available. The earlier [LionVision AI README](https://github.com/prasuldev/lion-face-classification/blob/abce141fda85a868121e6036a1037229f0bf35d7/README.md)
documents their binary checkpoint and dataset layout.

Created by Prasuldev.
