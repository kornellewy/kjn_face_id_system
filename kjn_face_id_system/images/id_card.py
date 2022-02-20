from pathlib import Path
from typing import Optional, List

from kjn_face_id_system.images.base_image import BaseKFISImage

class IdCaseKFISImage:
    def __init__(self, case_path: Path, image_name: str) -> None: