import re

import torch

from nougat import NougatModel
from nougat.utils.device import move_to_device
from nougat.utils.checkpoint import get_checkpoint
from nougat.postprocessing import markdown_compatible


class Nougat(object):
    def __init__(
        self,
    ):
        self.model = NougatModel.from_pretrained(
            get_checkpoint(model_tag="0.1.0-small")
        )
        self.model = move_to_device(
            self.model, bf16=True, cuda=True if torch.cuda.is_available() else False
        )
        self.model.eval()

    def __call__(self, image, cls=True):
        predictions = []
        model_output = self.model.inference(image=image, early_stopping=False)
        for j, output in enumerate(model_output["predictions"]):
            predictions.append(output)

        output = "".join(predictions).strip()
        output = re.sub(r"\n{3,}", "\n\n", output).strip()
        output = markdown_compatible(output)
        return output
