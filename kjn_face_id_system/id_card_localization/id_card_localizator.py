from typing import Optional
from pathlib import Path

from kjn_face_id_system.id_card_localization.detect import run
from kjn_face_id_system.images.bbox import BBox
from .models.common import DetectMultiBackend
from kjn_face_id_system.utils.utils import ID_CARD_CLASS_NAME

import torch


class IdCardLocalizator:
    def __init__(
        self,
        device: Optional[torch.device] = torch.device("cpu"),
    ) -> None:
        self.weights = "models/yolo5_localizator/weights/best.pt"
        self.data = "datasets/dataset_of_people_holding_id/data.yaml"
        self.device = device
        self.model = DetectMultiBackend(
            self.weights, device=self.device, dnn=False, data=self.data
        )

    def detect(self, image_path: Path) -> None:
        target_dir_path = image_path.parent
        line = run(
            model=self.model,
            weights=self.weights,
            data=self.data,
            source=target_dir_path,
            project=target_dir_path,
            device=self.device,
        )
        return BBox(
            image_path=image_path, yolo_line=line, bbox_class=ID_CARD_CLASS_NAME
        )
