from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import models
from torchvision.models import detection

CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
# Pin the revision that publishes the safetensors checkpoint.
CAPTION_REVISION = "a0ad1e4f8184ed95d6f80053778ca5b33aa31f5f"
CLASSIFIER_WEIGHTS = models.ResNet18_Weights.IMAGENET1K_V1
DETECTOR_WEIGHTS = detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
DETECTION_THRESHOLD = 0.45


def load_captioner():
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(CAPTION_MODEL, revision=CAPTION_REVISION)
    model = BlipForConditionalGeneration.from_pretrained(
        CAPTION_MODEL, revision=CAPTION_REVISION, use_safetensors=True
    )
    return model.eval(), processor


def load_classifier():
    return models.resnet18(weights=CLASSIFIER_WEIGHTS).eval()


def load_detector():
    return detection.ssdlite320_mobilenet_v3_large(weights=DETECTOR_WEIGHTS).eval()


@torch.inference_mode()
def describe_photo(captioner, photo: Image.Image) -> str:
    model, processor = captioner
    inputs = processor(images=photo, return_tensors="pt")
    tokens = model.generate(**inputs, max_new_tokens=40, num_beams=3)
    caption = processor.decode(tokens[0], skip_special_tokens=True).strip()
    if not caption:
        raise ValueError("The captioning model returned an empty description.")
    caption = caption[0].upper() + caption[1:]
    return caption if caption.endswith((".", "!", "?")) else caption + "."


@torch.inference_mode()
def identify_subject(classifier, photo: Image.Image) -> tuple[str, float]:
    batch = CLASSIFIER_WEIGHTS.transforms()(photo).unsqueeze(0)
    score, index = classifier(batch)[0].softmax(dim=0).max(dim=0)
    return CLASSIFIER_WEIGHTS.meta["categories"][index.item()], score.item()


@torch.inference_mode()
def find_objects(detector, photo: Image.Image) -> list[tuple]:
    predictions = detector([DETECTOR_WEIGHTS.transforms()(photo)])[0]
    keep = predictions["scores"] >= DETECTION_THRESHOLD
    categories = DETECTOR_WEIGHTS.meta["categories"]
    return [
        (box, categories[index], score)
        for box, index, score in zip(
            predictions["boxes"][keep].tolist(),
            predictions["labels"][keep].tolist(),
            predictions["scores"][keep].tolist(),
        )
    ]


def annotate_objects(photo: Image.Image, objects: list[tuple]) -> Image.Image:
    annotated = photo.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default(size=max(12, min(photo.size) // 32))
    color = "#FFD27E"
    for box, label, score in objects:
        left, top, _, _ = box
        draw.rectangle(box, outline=color, width=max(2, min(photo.size) // 140))
        text = f"{label} {score:.0%}"
        text_width = draw.textlength(text, font=font) + 8
        text_height = font.size + 8
        x = max(0, min(left, photo.width - text_width))
        y = max(0, top - text_height)
        draw.rectangle((x, y, x + text_width, y + text_height), fill=color)
        draw.text((x + 4, y + 4), text, font=font, fill="#171813", anchor="lt")
    return annotated
