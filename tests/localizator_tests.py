import unittest

from kjn_face_id_system.id_card_localization.detect import run


class LocalizatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.weights = 0

    def test_load(self):
        run(weights=self.weights)


if __name__ == "__main__":
    unittest.main()
