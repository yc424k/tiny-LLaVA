import copy
from dataclasses import dataclass
import json
from typing import Dict, Sequence, TYPE_CHECKING
from PIL import Image, ImageFile
import glob
import os

from .text_preprocess import TextPreprocess
from .image_preprocess import ImagePreprocess
from ..utils.arguments import DataArguments
from ..utils.constants import *


import transformers
import torch
from torch.utils.data import Dataset



ImageFile.LOAD_TRUNCATED_IMAGES = True

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)
        self.image_preprocess = ImagePreprocess(data_args.image_processor, data_args)
        self.sensor_field = getattr(data_args, 'sensor_field', None)
        if self.sensor_field is None and len(self.list_data_dict) > 0:
            if 'sensor' in self.list_data_dict[0]:
                self.sensor_field = 'sensor'
            elif 'sensor_data' in self.list_data_dict[0]:
                self.sensor_field = 'sensor_data'
        self.use_dummy_image = getattr(data_args, 'use_dummy_image', False)
        self.dummy_image_path = getattr(data_args, 'dummy_image_path', None)
        self._dummy_image_tensor = None

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        data_dict = self.text_preprocess(copy.deepcopy(sources["conversations"]))
        if 'image' in sources:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            image_path = os.path.join(image_folder, image_file)
            if not os.path.isfile(image_path):
                base, _ = os.path.splitext(image_path)
                candidates = sorted(glob.glob(base + '.*'))
                if candidates:
                    image_path = candidates[0]
                else:
                    raise FileNotFoundError(f"Image file not found: {image_path}")
            image = Image.open(image_path).convert('RGB')
            image = self.image_preprocess(image)
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            if self.use_dummy_image:
                data_dict['image'] = self._get_dummy_image_tensor()
            else:
                crop_size = getattr(self.data_args.image_processor, 'crop_size', getattr(self.data_args.image_processor, 'size'))
                if isinstance(crop_size, dict):
                    height = crop_size.get('height', 224)
                    width = crop_size.get('width', height)
                elif isinstance(crop_size, (list, tuple)):
                    if len(crop_size) >= 2:
                        height, width = crop_size[0], crop_size[1]
                    else:
                        height = width = crop_size[0]
                else:
                    height = width = crop_size if crop_size is not None else 224
                data_dict['image'] = torch.zeros(3, height, width)

        if self.sensor_field and self.sensor_field in sources:
            data_dict['sensor'] = copy.deepcopy(sources[self.sensor_field])
        return data_dict

    def _get_dummy_image_tensor(self):
        if self._dummy_image_tensor is not None:
            return self._dummy_image_tensor

        if self.dummy_image_path:
            dummy_image = Image.open(self.dummy_image_path).convert('RGB')
        else:
            dummy_image = self._create_blank_image()

        self._dummy_image_tensor = self.image_preprocess(dummy_image)
        return self._dummy_image_tensor

    def _create_blank_image(self):
        processor = getattr(self.image_preprocess, 'image_processor', None)
        width = height = 224
        if processor is not None:
            crop_size = getattr(processor, 'crop_size', None)
            if isinstance(crop_size, dict):
                height = crop_size.get('height', crop_size.get('shortest_edge', height))
                width = crop_size.get('width', crop_size.get('shortest_edge', height))
            elif isinstance(crop_size, (list, tuple)):
                if len(crop_size) >= 2:
                    height, width = crop_size[0], crop_size[1]
                elif len(crop_size) == 1:
                    height = width = crop_size[0]
            else:
                size = getattr(processor, 'size', None)
                if isinstance(size, dict):
                    height = size.get('height', size.get('shortest_edge', height))
                    width = size.get('width', size.get('shortest_edge', height))
                elif isinstance(size, (list, tuple)):
                    if len(size) >= 2:
                        height, width = size[0], size[1]
                    elif len(size) == 1:
                        height = width = size[0]
                elif isinstance(size, int):
                    height = width = size

        width = int(width)
        height = int(height)
        return Image.new('RGB', (max(width, 1), max(height, 1)), color=(0, 0, 0))


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels[:, :self.tokenizer.model_max_length]
        # FIXME: This is a hack for handling phi and stablelm, as they have the same eos, pad and unk. We want the model
        # FIXME: to predict the eos in the input ids, but we also use the id of eos to pad sequence, so we use a temp
        # FIXME: eos id first, and convert them back.
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        if any('sensor' in instance for instance in instances):
            sensors = [instance.get('sensor') for instance in instances]
            if sensors and torch.is_tensor(sensors[0]):
                lengths = []
                tensor_sensors = []
                for sensor in sensors:
                    if sensor is None:
                        sensor = torch.zeros(1, dtype=torch.float32)
                    if sensor.dtype != torch.float32:
                        sensor = sensor.to(torch.float32)
                    tensor_sensors.append(sensor)
                    lengths.append(sensor.shape[0])
                padded_sensors = torch.nn.utils.rnn.pad_sequence(tensor_sensors, batch_first=True)
                sensor_mask = torch.zeros(padded_sensors.shape[:2], dtype=torch.bool)
                for idx, length in enumerate(lengths):
                    if length > 0:
                        sensor_mask[idx, :length] = True
                batch['sensors'] = padded_sensors
                batch['sensor_mask'] = sensor_mask
            else:
                batch['sensors'] = sensors

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
