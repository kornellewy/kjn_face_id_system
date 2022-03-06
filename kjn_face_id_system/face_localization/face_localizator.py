from pathlib import Path
from PIL import Image
from typing import Optional, Tuple
import numpy as np

import torch
from facenet_pytorch import MTCNN

from kjn_face_id_system.images.bbox import BBox
from kjn_face_id_system.utils.utils import (
    FACE_CLASS_NAME,
)


class FaceLocalizator:
    def __init__(
        self,
        device: Optional[torch.device] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        self.device = device
        self.model = MTCNN(
            keep_all=True, device=self.device, thresholds=[0.5, 0.6, 0.6]
        )

    def detect(self, image_path: Path) -> Tuple[list, list, list]:
        image = Image.open(image_path.as_posix())
        width, height = image.size
        boxes, prob = self.model.detect(image)
        output_bboxes = []
        if not isinstance(boxes, np.ndarray):
            return output_bboxes
        prob = prob.tolist()
        boxes = boxes.tolist()
        for idx, bbox in enumerate(boxes):
            x_top_left = int(bbox[0]) / width
            y_top_left = int(bbox[1]) / height
            w = (int(bbox[2]) - int(bbox[0])) / width
            h = (int(bbox[3]) - int(bbox[1])) / height
            x_center = x_top_left + (w / 2)
            y_center = y_top_left + (h / 2)
            line = [0, x_center, y_center, w, h, prob[idx]]
            output_bboxes.append(
                BBox(image_path=image_path, yolo_line=line, bbox_class=FACE_CLASS_NAME)
            )
        return output_bboxes
