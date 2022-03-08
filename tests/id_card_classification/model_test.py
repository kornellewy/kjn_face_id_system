import unittest

import torch

from kjn_face_id_system.id_card_classification.model import Siamese


class IdCaseKFISImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = Siamese()

    def test_forward(self):
        input_tensor = torch.rand([1, 3, 224, 224])
        output_tensor = self.model.forward(input_tensor, input_tensor)
        self.assertEqual(output_tensor.shape, torch.Size([1, 1]))


if __name__ == "__main__":
    unittest.main()
