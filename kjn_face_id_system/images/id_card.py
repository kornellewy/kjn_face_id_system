from pathlib import Path
from typing import Optional, List

from kjn_face_id_system.images.base_image import BaseKFISImage


class KFISIdCase(BaseKFISImage):
    def __init__(
        self, image_path: Path, case_path: Path, tags: Optional[dict] = None
    ) -> None:
        super().__init__(image_path=image_path)
        self.case_path = case_path
        self.tags = tags
