# app.py v11あり
import os
from tabnanny import verbose
from pathlib import Path
from typing import List, Tuple, Literal, Dict, Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import cv2
import numpy as np
import albumentations as A
from mmdet.apis import init_detector, inference_detector
import matplotlib.pyplot as plt
import io
import torch
from PIL import Image
from torch import nn
from pydantic import BaseModel
from mmdet.apis import init_detector, inference_detector
from ultralytics import YOLO

from component.yolov11_arts import YOLOv11Detector
# リクエストボディの定義
class DetectionRequest(BaseModel):
    classname: Literal["stop", "sl65","All"]
    file: str

app = FastAPI()


def _safe_init_detector(label: str, config_path: str, checkpoint_path: str, device: str):
    try:
        return init_detector(config_path, checkpoint_path, device=device)
    except Exception as exc:
        print(f"[warn] Failed to load Faster R-CNN model '{label}': {exc}")
        return None


def _safe_torch_hub_load(label: str, repo_or_dir: str, model: str, **kwargs):
    try:
        return torch.hub.load(repo_or_dir, model, **kwargs)
    except Exception as exc:
        print(f"[warn] Failed to load YOLO model '{label}': {exc}")
        return None


def _safe_yolov11_detector(label: str, weights_path: str):
    try:
        return YOLOv11Detector(weights_path)
    except Exception as exc:
        print(f"[warn] Failed to load YOLOv11 model '{label}': {exc}")
        return None

# initialize models
fasterrcnn_models = {
    "arts": _safe_init_detector(
        "arts",
        "<PATH_TO_ARTS_CONFIG>",
        "<PATH_TO_ARTS_CHECKPOINT>",
        device="cpu",
    ),
    "mapillary": _safe_init_detector(
        "mapillary",
        "<PATH_TO_MAPILLARY_CONFIG>",
        "<PATH_TO_MAPILLARY_CHECKPOINT>",
        device="cpu",
    ),
}
fasterrcnn_models = {k: v for k, v in fasterrcnn_models.items() if v is not None}

ground_truth_dict = {
    "arts": {
        "stop": 31,
        "sl65": 41
    },
    "mapillary": {
        "stop": 7,
        "sl65": 6
    },
    "coco":{
        "stop": 11,
        "sl65": None
    }
}

yolo_models = {
    "arts": _safe_torch_hub_load(
        "arts",
        "ultralytics/yolov5",
        "custom",
        path="best-1008.pt",
        verbose=True,
    ),
    "coco": _safe_torch_hub_load("coco", "ultralytics/yolov5", "yolov5s", verbose=True),
}
yolo_models = {k: v for k, v in yolo_models.items() if v is not None}

yolo_v11 = {
    "arts": _safe_yolov11_detector(
        "arts", "<PATH_TO_YOLOV11_WEIGHTS>"
    )
}
yolo_v11 = {k: v for k, v in yolo_v11.items() if v is not None}


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIMPLECNN_MODEL_NAME = "SimpleCNN"
SIMPLECNN_IMAGE_SIZE = 60
SIMPLECNN_MODEL_DIR = Path(
    os.environ.get(
        "SIMPLECNN_MODEL_DIR",
        "../Classifier/models_60",
    )
)

SIMPLECNN_DATASET_ALIASES = {
    "arts": "ARTS",
    "gtsrb": "GTSRB",
    "lisa": "LISA",
    "mapillary": "Mapillary",
}

SIMPLECNN_STOP_SIGN_INDEX_BY_DATASET = {
    "LISA": 0,
    "GTSRB": 14,
    "ARTS": 12,
    "Mapillary": 6,
}

SIMPLECNN_TRANSFORM = A.Compose([
    A.Resize(SIMPLECNN_IMAGE_SIZE, SIMPLECNN_IMAGE_SIZE),
    A.Normalize(),
])


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, image_size: int = SIMPLECNN_IMAGE_SIZE) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
        )

        # derive flattened feature size dynamically to avoid magic numbers
        with torch.no_grad():
            example = torch.zeros(1, 3, image_size, image_size)
            flattened_dim = self.features(example).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits


_simplecnn_cache = {}


def _resolve_simplecnn_dataset(dataset: str) -> str:
    key = dataset.strip().lower()
    if key not in SIMPLECNN_DATASET_ALIASES:
        raise KeyError(f"Unsupported dataset '{dataset}'")
    return SIMPLECNN_DATASET_ALIASES[key]


def _normalize_simplecnn_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    if any(key.startswith("net.") for key in state_dict):
        return _remap_simplecnn_net_state_dict(state_dict)
    return state_dict


def _remap_simplecnn_net_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        parts = key.split(".")
        if len(parts) < 3 or parts[0] != "net":
            remapped[key] = value
            continue
        try:
            layer_idx = int(parts[1])
        except ValueError:
            remapped[key] = value
            continue
        suffix = ".".join(parts[2:])
        if layer_idx <= 11:
            new_key = f"features.{layer_idx}.{suffix}"
        elif layer_idx == 13:
            new_key = f"classifier.0.{suffix}"
        elif layer_idx == 15:
            new_key = f"classifier.2.{suffix}"
        elif layer_idx == 17:
            new_key = f"classifier.4.{suffix}"
        else:
            new_key = key
        remapped[new_key] = value
    return remapped


def _load_simplecnn_model(model_path: Path) -> Tuple[nn.Module, List[str]]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    classes = list(checkpoint["classes"])
    model = SimpleCNN(num_classes=len(classes))
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = _normalize_simplecnn_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model, classes


def _get_simplecnn_model(dataset_name: str) -> Tuple[nn.Module, List[str]]:
    cached = _simplecnn_cache.get(dataset_name)
    if cached is not None:
        return cached
    model_path = SIMPLECNN_MODEL_DIR / f"{SIMPLECNN_MODEL_NAME}_{dataset_name}_best_loss.pth"
    model, classes = _load_simplecnn_model(model_path)
    _simplecnn_cache[dataset_name] = (model, classes)
    return model, classes


def _preprocess_simplecnn_array(image: np.ndarray) -> torch.Tensor:
    processed = SIMPLECNN_TRANSFORM(image=image)["image"]
    tensor = torch.from_numpy(processed.transpose(2, 0, 1)).float()
    return tensor.unsqueeze(0)


def _crop_stop_sign_with_yolo(image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
    model = yolo_models.get("coco")
    if model is None:
        print("[warn] YOLO COCO model is not initialized")
        return None, None

    model.conf = 0.01
    results = model(image)
    detections = results.pandas().xyxy[0]
    stop_detections = detections[detections["name"] == "stop sign"]
    if len(stop_detections) == 0:
        return None, None

    best = stop_detections.sort_values("confidence", ascending=False).iloc[0]
    x_min = max(int(best["xmin"]), 0)
    y_min = max(int(best["ymin"]), 0)
    x_max = min(int(best["xmax"]), image.shape[1])
    y_max = min(int(best["ymax"]), image.shape[0])
    if x_max <= x_min or y_max <= y_min:
        return None, float(best["confidence"])

    return image[y_min:y_max, x_min:x_max], float(best["confidence"])


@app.post("/detect/yolo_v11/{dataset}")
async def detect_yolov11(dataset: str, request: DetectionRequest):
    if dataset not in yolo_v11:
        return JSONResponse(content={"error": "Dataset not found"}, status_code=404)

    model = yolo_v11[dataset]
    model.model.conf = 0.01
    
    classid = ground_truth_dict[dataset][request.classname]
    filter_class_name = model.model.names[classid]
    print(filter_class_name)
    result = model.predict(image_path = request.file,conf = 0.01)
    print(result)
    
    if len(result) ==0:
        print("No Sign Detected (Not filterone, but all classes)")
        return 0.0
    else:
        result_filter_by_class = result.filter_by_class(filter_class_name)
        
        if len(result_filter_by_class)==0:
            return 0
        
        # conf 順にソート
        best_frame = result_filter_by_class.sort_values('confidence', ascending=False).iloc[0]
        
        return best_frame["confidence"]
    
    



@app.post("/detect/fasterrcnn/{dataset}")
async def detect_fasterrcnn(dataset: str,request:DetectionRequest):
    if dataset not in fasterrcnn_models:
        return JSONResponse(content={"error": "Dataset not found"}, status_code=404)

    model = fasterrcnn_models[dataset]
    

    
    img = cv2.imread(request.file)

    result = inference_detector(model, img)
    
    print(result)
#
    
    # you can access the predicted class through result[class_id]
    classid: int = ground_truth_dict[dataset][request.classname]
    if len(result) == 0:
        return 0
    probs = result[classid]

    if len(probs) > 0:
        p = probs[:, 4].max()
    else:
        p = 0
    return float(p)

@app.post("/detect/yolo/{dataset}")
async def detect_yolo(dataset: str, request:DetectionRequest):
    
    if dataset not in yolo_models:
        return JSONResponse(content={"error": "Dataset not found"}, status_code=404)

    model = yolo_models[dataset]
    model.conf = 0.01
    print(request.file)
    result = model(request.file)
    
    
    if dataset == "arts":
        if request.classname == "stop":
            filter_name = "R1-1"
        elif request.classname == "sl65":
            filter_name = "R2-165"
        elif request.classname == "All":
            detection = result.pandas().xyxy[0]
            if len(detection) == 0 :
                return 0.0
            else:
                return detection['confidence'].max()
        else:
            return JSONResponse(content={"error": "Class not found"}, status_code=404)
    elif dataset == "coco":
        if request.classname == "stop":
            filter_name = "stop sign"
        
        
            
        else:
            return JSONResponse(content={"error": "Class not found"}, status_code=404)
    # R1-1のconfidenceを取得
    detection = result.pandas().xyxy[0]
    print(detection)
    class_detections = detection[detection['name'] == filter_name]
    if len(class_detections) == 0:
        return 0 
            
    else:
        
        max_confidence = class_detections['confidence'].max()
        return float(max_confidence)

@app.post("/detect/SimpleCNN/{dataset}")
async def detect_simplecnn(dataset: str, request: DetectionRequest):
    try:
        dataset_name = _resolve_simplecnn_dataset(dataset)
    except KeyError:
        return JSONResponse(content={"error": "Dataset not found"}, status_code=404)

    if request.classname not in {"stop", "All"}:
        return JSONResponse(content={"error": "Class not found"}, status_code=404)

    try:
        model, classes = _get_simplecnn_model(dataset_name)
        image_path = Path(request.file)
        image = np.array(Image.open(image_path).convert("RGB"))
        if request.classname == "stop":
            cropped, _ = _crop_stop_sign_with_yolo(image)
            if cropped is None:
                return float(0.0)
            image = cropped
        image_batch = _preprocess_simplecnn_array(image).to(DEVICE)
    except FileNotFoundError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=404)

    with torch.no_grad():
        probabilities = model(image_batch).softmax(dim=1)

    top_prob, top_index = probabilities.max(dim=1)
    top_class = classes[int(top_index.item())]
    if request.classname == "All":
        return {
            "top_class": top_class,
            "top_prob": float(top_prob.item()),
        }

    stop_index = SIMPLECNN_STOP_SIGN_INDEX_BY_DATASET[dataset_name]
    if stop_index >= len(classes):
        return JSONResponse(content={"error": "Stop class index out of range"}, status_code=500)

    stop_prob = float(probabilities[0, stop_index].item())
    correctly_classified = int(top_index.item()) == stop_index
    if not correctly_classified:
        
        return float(0.0)
    
    return float(stop_prob)
        
    
    """
    return {
        "prob": stop_prob,
        "correctly_classified": correctly_classified,
        "top_class": top_class,
        "top_prob": float(top_prob.item()),
    }
    """

@app.get("/models")
async def list_models():
    return {
        "fasterrcnn_models": list(fasterrcnn_models.keys()),
        "yolo_models": list(yolo_models.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
