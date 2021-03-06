import os
from operator import itemgetter
import numpy as np
from typing import List, Tuple

import cv2
import torch
from easyocr import Reader


class KjnOcr(object):
    def __init__(
        self,
        languages: list = ["pl"],
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        super().__init__()
        self.languages = languages
        self.device = device
        self.white_list = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZśŚćĆźŹóÓżŻęĘąĄłŁŃń0123456789"
        if str(self.device) == "cuda":
            self.gpu = True
        else:
            self.gpu = False
        self.reader = Reader(self.languages, gpu=self.gpu)

    def read(self, img: np.ndarray) -> Tuple[List[dict], List[str]]:
        results = self.reader.readtext(img, batch_size=64, allowlist=self.white_list)
        textes_with_bbox = []
        textes = []
        for (bbox, text, prob) in results:
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))
            textes.append(text)
            bbox_dict = {
                "top_left": tl,
                "top_right": tr,
                "botom_right": br,
                "botom_left": bl,
                "text": text,
            }
            textes_with_bbox.append(bbox_dict)
        return textes_with_bbox, textes

    # def read_ranked_by_hight(self, img):
    #     # text ranked by hight of box
    #     results = self.reader.readtext(img, batch_size=64, allowlist=self.white_list)
    #     textes_with_bbox = []
    #     textes = []
    #     for (bbox, text, prob) in results:
    #         (tl, tr, br, bl) = bbox
    #         tl = (int(tl[0]), int(tl[1]))
    #         tr = (int(tr[0]), int(tr[1]))
    #         br = (int(br[0]), int(br[1]))
    #         bl = (int(bl[0]), int(bl[1]))
    #         textes.append(text)
    #         hight = bl[1] - tl[1]
    #         bbox_dict = {
    #             "hight": hight,
    #             "top_left": tl,
    #             "top_right": tr,
    #             "botom_right": br,
    #             "botom_left": bl,
    #             "text": text,
    #         }
    #         textes_with_bbox.append(bbox_dict)
    #     textes_with_bbox = sorted(textes_with_bbox, key=itemgetter("hight"))
    #     textes_with_bbox.reverse()
    #     return textes_with_bbox

    def read_and_draw(self, img):
        results = self.reader.readtext(img, batch_size=64)
        img_with_bbox_and_text = img
        textes_with_bbox = []
        textes = []
        for (bbox, text, prob) in results:
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))
            textes.append(text)
            hight = bl[1] - tl[1]
            bbox_dict = {
                "hight": hight,
                "top_left": tl,
                "top_right": tr,
                "botom_right": br,
                "botom_left": bl,
                "text": text,
            }
            textes_with_bbox.append(bbox_dict)
            cv2.rectangle(img_with_bbox_and_text, tl, br, (0, 255, 0), 2)
            cv2.putText(
                img_with_bbox_and_text,
                str(text),
                (tl[0], tl[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
        return textes_with_bbox, img_with_bbox_and_text, textes

    def cleanup_text(self, text):
        return "".join([c if ord(c) < 128 else "" for c in text]).strip()


if __name__ == "__main__":

    def load_images(path):
        images = []
        valid_images = [".png", ".jpg", ".jpeg"]
        for f in os.listdir(path):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            images.append(os.path.join(path, f))
        return images

    import pprint

    kjn = KjnOcr(device = torch.device("cpu"))
    print("kjn.device: ", kjn.device)
    print("kjn.gpu: ", kjn.gpu)
    image = cv2.imread("polish_id_cards_dataset/after_7_11_2021/1_front.png")
    textes_with_bbox, img_with_bbox_and_text, textes = kjn.read_and_draw(image)
    pprint.pprint(textes_with_bbox)
    cv2.imshow("img", img_with_bbox_and_text)
    cv2.waitKey(0)
