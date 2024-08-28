import re
from tqdm import tqdm
from typing import Callable
from functools import partial

import torch
from torch.utils.data import Dataset

from nougat.model import NougatModel
from nougat.utils.device import move_to_device
from nougat.utils.checkpoint import get_checkpoint
from nougat.postprocessing import markdown_compatible


class NougatDataset(Dataset):
    def __init__(self, name, prepare_fn: Callable, image_list: list):
        super().__init__()
        self.name = name
        self.prepare_fn = prepare_fn
        self.image_list = image_list
        self.size = len(self.image_list)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if i <= self.size and i >= 0:
            return self.prepare_fn(self.image_list[i]), (
                self.name if i == self.size - 1 else ""
            )
        else:
            raise IndexError

    @staticmethod
    def ignore_none_collate(batch):
        if batch is None:
            return None, None
        try:
            _batch = []
            for i, x in enumerate(batch):
                image, name = x
                if image is not None:
                    _batch.append(x)
                elif name:
                    if i > 0:
                        _batch[-1] = (_batch[-1][0], name)
                    elif len(batch) > 1:
                        _batch.append((batch[1][0] * 0, name))
            if len(_batch) == 0:
                return None, None
            return torch.utils.data.dataloader.default_collate(_batch)
        except AttributeError:
            pass
        return None, None


class Nougat(object):
    def __init__(self, batchsize=10):
        self.model = NougatModel.from_pretrained(
            get_checkpoint(model_tag="0.1.0-small")
        )
        self.model = move_to_device(
            self.model, bf16=True, cuda=True if torch.cuda.is_available() else False
        )
        self.model.eval()
        self.name = "nougat"
        self.model.decoder.model.prepare_inputs_for_generation = self.prepare_inputs_for_inference
        self.batchsize = batchsize
    
    def prepare_inputs_for_inference(
        self,
        input_ids: torch.Tensor,
        encoder_outputs: torch.Tensor,
        past=None,
        past_key_values=None,
        use_cache: bool = None,
        attention_mask: torch.Tensor = None,
        cache_position=None,
    ):
        """
        Args:
            input_ids: (batch_size, sequence_length)

        Returns:
            input_ids: (batch_size, sequence_length)
            attention_mask: (batch_size, sequence_length)
            encoder_hidden_states: (batch_size, sequence_length, embedding_dim)
        """
        attention_mask = input_ids.ne(self.model.decoder.tokenizer.pad_token_id).long()
        past = past or past_key_values
        if past is not None:
            input_ids = input_ids[:, -1:]
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past,
            "use_cache": use_cache,
            "encoder_hidden_states": encoder_outputs.last_hidden_state,
        }
        return output

    def __call__(self, images: list):
        dataset = NougatDataset(
            self.name,
            partial(self.model.encoder.prepare_input, random_padding=False),
            images,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batchsize,
            shuffle=False,
            collate_fn=NougatDataset.ignore_none_collate,
        )

        predictions = []
        for i, (sample, _) in enumerate(tqdm(dataloader)):
            model_output = self.model.inference(
                image_tensors=sample, early_stopping=True
            )
            for j, output in enumerate(model_output["predictions"]):
                if output.strip() == "[MISSING_PAGE_POST]":
                    predictions.append("[MISSING_PAGE]")
                elif model_output["repeats"][j] is not None:
                    if model_output["repeats"][j] > 0:
                        predictions.append("[MISSING_PAGE]")
                    else:
                        predictions.append("[MISSING_PAGE]")
                else:
                    output = output.strip()
                    output = re.sub(r"\n{3,}", "\n\n", output).strip()
                    output = markdown_compatible(output)
                    predictions.append(output)

        return predictions
