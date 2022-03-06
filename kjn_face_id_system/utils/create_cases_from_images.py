from pathlib import Path
import shutil

from kjn_face_id_system.utils.utils import (
    ID_CARD_BACK_DIR_NAME,
    ID_CARD_FRONT_DIR_NAME,
    SELFIE_DIR_NAME,
    IMAGE_NAME,
)


class CaseCreatorFromImages:
    def __init__(self) -> None:
        self.selfe_images_paths = [
            "datasets/my_images/20210616_101318.png",
            "datasets/my_images/20220306_091259.png",
            "datasets/my_images/20220306_091351.png",
            "datasets/my_images/20220306_091353.png",
        ]
        self.front_images_paths = ["datasets/my_images/20210121_105253.png"]
        self.back_images_paths = [
            "datasets/my_images/20210121_105307.png",
            "datasets/my_images/20220306_091227.png",
            "datasets/my_images/20220306_091230.png",
            "datasets/my_images/20220306_091234.png",
        ]
        self.dir_names_paths = (
            SELFIE_DIR_NAME,
            ID_CARD_FRONT_DIR_NAME,
            ID_CARD_BACK_DIR_NAME,
        )

    def get_last_idex(self, dir_path: Path) -> int:
        indexes = [int(file.stem) for file in dir_path.iterdir() if file.is_dir()]
        return max(indexes) + 1

    def all_combinations(self) -> list:
        all_combinations = []
        for selfe_image_path in self.selfe_images_paths:
            for front_image_path in self.front_images_paths:
                for back_image_path in self.back_images_paths:
                    combination = (selfe_image_path, front_image_path, back_image_path)
                    all_combinations.append(combination)
        return all_combinations

    def create_output_paths(
        self, output_dir_path: Path, images_paths: tuple, idex: int
    ) -> tuple:
        output_paths = []
        for idx, image_path in enumerate(images_paths):
            output_path = output_dir_path.joinpath(
                str(idex), self.dir_names_paths[idx], IMAGE_NAME
            )
            Path(output_path.parent).mkdir(exist_ok=True, parents=True)
            output_paths.append(output_path)
        return tuple(output_paths)

    def copy_images(self, input_paths: tuple, output_paths: tuple) -> None:
        for input_path, output_path in zip(input_paths, output_paths):
            shutil.copy2(input_path, output_path)

    def create_examples(self, output_dir_path: Path) -> None:
        first_index = self.get_last_idex(output_dir_path)
        all_combinations = self.all_combinations()
        for idx, combination in enumerate(all_combinations):
            idx = first_index + idx
            output_paths = self.create_output_paths(
                output_dir_path=output_dir_path, images_paths=combination, idex=idx
            )
            self.copy_images(combination, output_paths)
        pass


if __name__ == "__main__":
    combinations_paths = Path("datasets/combinations")
    creator = CaseCreatorFromImages()
    print(creator.create_examples(combinations_paths))
