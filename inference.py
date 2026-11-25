from pathlib import Path

from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision import models, transforms

# ImageFolder sorts the original training folders: lions, not_lions.
CLASS_LABELS = ("Lion", "Not a lion")
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def read_photo(source):
    with Image.open(source) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def load_model(checkpoint: Path):
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_LABELS))
    model.load_state_dict(weights)
    return model.eval()


@torch.inference_mode()
def classify(model, photo: Image.Image) -> tuple[str, float]:
    batch = PREPROCESS(photo).unsqueeze(0)
    scores = model(batch).softmax(dim=1)
    confidence, index = scores[0].max(dim=0)
    return CLASS_LABELS[index.item()], confidence.item()
