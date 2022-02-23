from pathlib import Path
from typing import Optional


class BBox:
    def __init__(
        self,
        image_path: Path,
        line: list,
        bbox_class: str,
        tags: Optional[dict] = None,
    ) -> None:
        self.image_path = image_path
        self.bbox_class = bbox_class
        self.tags = tags
        (
            self.x_center,
            self.y_center,
            self.width,
            self.height,
            self.confidence,
        ) = self.preproces_line(line=line)

    def __repr__(self):
        return f"{self.bbox_class}-{self.x_center}-{self.y_center}-{self.width}-{self.height}-{self.confidence}"

    def preproces_line(self, line: list) -> dict:
        line = [round(float(num), 2) for num in line]
        return line[1], line[2], line[3], line[4], line[5]

    def get_coco(self):
        x_top_left = self.x_center - self.width / 2
        y_top_left = self.y_center - self.height / 2
        return {
            "bbox_class": self.bbox_class,
            "x_top_left": x_top_left,
            "y_top_left": y_top_left,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
        }
