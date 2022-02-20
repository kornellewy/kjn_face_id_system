from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class BaseKFISImage:
    def __init__(self, image_path: Path, tags: Optional[dict]=None) -> None:
        super().__init__()
        self.image_path = image_path
        self.image_name = image_path.name
        self.tags = tags
        self.image = cv2.imread(image_path.as_posix())
        self.height, self.width = self.get_image_width_height()

    def get_image_width_height(self) -> np.ndarray:
        return self.image.shape[:2]

    def save_image(
        self,
        new_image_path: Path,
        grayscale: Optional[bool] = False,
        update: Optional[bool] = False,
    ) -> None:
        if grayscale:
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            image = self.image
        if update:
            self.image_path = new_image_path
        return cv2.imwrite(new_image_path.as_posix(), image)

    def resize_image(self, new_width: int, new_height: int) -> None:
        if new_width > self.width or new_height > self.height:
            self.uplarge_image(new_width=new_width, new_height=new_height)
        else:
            self.shrink_image(new_width=new_width, new_height=new_height)
        self.height, self.width = self.get_image_width_height()

    def uplarge_image(self, new_width: int, new_height: int) -> None:
        self.image = cv2.resize(
            self.image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )

    def shrink_image(self, new_width: int, new_height: int) -> None:
        self.image = cv2.resize(
            self.image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
