import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import os

def obj_detection(image,candidate_labels):
    print(candidate_labels)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model_id = r"/home/mll/下载/models--IDEA-Research--grounding-dino-base/snapshots/12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    inputs = processor(images=image, text=candidate_labels, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )
    print("results:",results)
    return results


