from typing import Optional
from pathlib import Path

from kjn_face_id_system.id_card_localization.detect import run
from .models.common import DetectMultiBackend


class IdCardLocalizator:
    def __init__(self, device: Optional[str] = "cpu") -> None:
        self.weights = "models/yolo5_localizator/weights/best.pt"
        self.data = "datasets/dataset_of_people_holding_id/data.yaml"
        self.device = device
        self.model = DetectMultiBackend(self.weights, device=device, dnn=False, data=self.data)
        
    def detect(self, image_path: Path) -> None:
        target_dir_path = image_path.parent
        line = run(
            model = self.model,
            weights=self.weights,
            data=self.data,
            source=target_dir_path,
            project=target_dir_path,
            device=self.device
        )
        return self.preproces_line(line)

    def preproces_line(self, line: list) -> dict:
        line = [float(num) for num in line]
        return{
            "class_index": (line[0]),
            "x_center": line[1],
            "y_center": line[2],
            "width": line[3],
            "height": line[4],
            "confidence": line[5],
        }
            
                

        