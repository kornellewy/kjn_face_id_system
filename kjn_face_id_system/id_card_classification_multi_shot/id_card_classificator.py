# ######################## EXAMPLE USAGE###########################
# import torch
# from PIL import Image
# import torch.nn as nn
# import torchvision
# from torchvision import models, transforms
# import numpy as np

# # def your image path
# image_path = ''

# # def your model path can be relative can be apsolute
# model_path = ''

# # def transforms
# transforms = transforms.Compose([
#             transforms.Resize(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406],
#                                  [0.229, 0.224, 0.225])
#         ])

# # check avalible device
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # load model
# model = models.resnet50(pretrained=False).to(device)
# for param in model.parameters():
#     param.requires_grad = False
# model.fc = nn.Sequential(
#                nn.Linear(2048, 128),
#                nn.ReLU(inplace=True),
#                nn.Linear(128, 2)).to(device)
# model.load_state_dict(torch.load(model_path), strict=False)
# model = model.to(device)
# model.eval()

# # laod image
# image = Image.open(image_path)
# image_tensor = transforms(image)
# image_tensor = image_tensor.unsqueeze(0)
# image_tensor = image_tensor.to(device)
# logps = model(image_tensor)
# ps = torch.exp(logps)
# top_p, top_class = ps.topk(1, dim=1)
# index = top_class.item()

# ############################# INPUT #############################
# image size: (224, 244, 3)

# ############################ OUTPUT #############################
# int value: 0 or 1

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from kjn_face_id_system.id_card_classification_multi_shot.classification_universal_trainscript import (
    UniversaClassificationTrainer,
)


class IdCardClassificator:
    WEIGHT_PATH = Path("models/id_card_classification_multi_shot/models/resnet50.pth")
    CLASS_NAME_TO_IDX = {
        "albanian_front": 0,
        "argentine_front": 1,
        "austria_back": 2,
        "austria_front": 3,
        "belgium_back": 4,
        "belgium_front": 5,
        "bosnian_front": 6,
        "bulgaria_back": 7,
        "bulgaria_front": 8,
        "chile_front": 9,
        "croatia_back": 10,
        "croatia_front": 11,
        "czech_back": 12,
        "czech_front": 13,
        "estonia_back": 14,
        "estonia_front": 15,
        "finland_back": 16,
        "finland_front": 17,
        "france_back": 18,
        "france_front": 19,
        "germany_back": 20,
        "germany_front": 21,
        "hungary_type_1_back": 22,
        "hungary_type_1_front": 23,
        "hungary_type_2_front": 24,
        "indonesian_front": 25,
        "israel_front": 26,
        "italian_back": 27,
        "italian_front": 28,
        "latvia_back": 29,
        "latvia_front": 30,
        "liechtenstein_back": 31,
        "liechtenstein_front": 32,
        "lithuania_back": 33,
        "lithuania_front": 34,
        "macau_back": 35,
        "macau_front": 36,
        "malta_front": 37,
        "norway_back": 38,
        "norway_front": 39,
        "poland_type_1_front": 40,
        "poland_type_2_front": 41,
        "poland_type_3_front": 42,
        "poland_type_4_back": 43,
        "poland_type_4_front": 44,
        "portugal_front": 45,
        "romania_back": 46,
        "romania_front": 47,
        "serbian_front": 48,
        "slovakia_back": 49,
        "slovakia_front": 50,
        "south_africa_front": 51,
        "sweden_back": 52,
        "sweden_front": 53,
        "turkish_front": 54,
        "ukrainian_front": 55,
        "uruguay_front": 56,
        "usa_passport_back": 57,
        "usa_passport_front": 58,
        "venezuela_front": 59,
    }
    IDX_TO_CLASS_NAME = {i: c for c, i in CLASS_NAME_TO_IDX.items()}

    def __init__(
        self,
        device: Optional[torch.device] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        self.device = device
        self.img_transform = A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        self.model = torch.load(self.WEIGHT_PATH)
        self.model.to(self.device)

    def predict(self, image: np.ndarray) -> dict:
        image = self.img_transform(image=image)["image"]
        image = image.unsqueeze(0)
        image.to(self.device)
        with torch.no_grad():
            prediction = self.model(image)
        top_p, top_class = prediction.topk(1, dim=1)
        top_class = top_class.item()
        top_p = top_p.item()
        class_name = self.IDX_TO_CLASS_NAME[top_class]
        return {
            "prediction": prediction,
            "top_p": top_p,
            "top_class": top_class,
            "class_name": class_name,
        }
