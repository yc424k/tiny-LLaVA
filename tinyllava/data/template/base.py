from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
import copy
import re

from .formatter import EmptyFormatter, StringFormatter
from .formatter import Formatter
from ...utils.constants import *

from transformers import PreTrainedTokenizer
import torch
    


@dataclass
class Template:
    format_image_token: "Formatter"
    format_user: "Formatter"
    format_assistant: "Formatter"
    system: "Formatter"
    separator: "Formatter"
    format_sensor_token: "Formatter" = field(default_factory=lambda: StringFormatter(slot=f"{DEFAULT_SENSOR_TOKEN}\n{{content}}"))
    
    def encode(self, messages, tokenizer, mode='train'):
        """
        1. get list form messages(conversations:[{from:human, value:message}, {from:gpt, value:message}])
            ===>  human_list, value_list
        2. prompt two list
        3. tokenize prompt
        4. make target
        """
        question_list, answer_list = self.get_list_from_message(messages)
        prompt = self.prompt(question_list, answer_list)
        input_ids = self.tokenizer_image_token(prompt, tokenizer, return_tensors='pt')
        if mode == 'train':
            labels = self.make_labels(input_ids, prompt, tokenizer)
            return dict(
                input_ids=input_ids,
                labels=labels
            )
        else:
            return dict(input_ids=input_ids, prompt=prompt)
        
    
    def get_list_from_message(self, messages):
        return self._get_list_from_message(messages)
    
    def _get_list_from_message(self, messages):
        """
        messages  ====>  [{from:human, value:message}, {from:gpt, value:message}]
        """
        question_list = []
        answer_list = []
        first_is_not_question = 0
        for i, message in enumerate(messages):
            if i == 0 and message['from'] != 'human':
                first_is_not_question = 1
                continue
            if i % 2 == first_is_not_question:
                question_list.append(message['value'])
            else:
                answer_list.append(message['value'])
        
        assert len(question_list) == len(answer_list) , \
            f"qa is not match : length_q:{len(question_list)} vs length_a:{len(answer_list)}"
        return question_list, answer_list
    

    def prompt(
        self,
        question_list, answer_list
    ):
        if type(question_list) is str:
            question_list = [question_list]
        if type(answer_list) is str:
            answer_list = [answer_list]    
        msg = self._prompt(question_list, answer_list)
        return msg

    def _prompt(
        self,
        question_list, answer_list,
    ):
        msg = ""
        for i, (question, answer) in enumerate(zip(question_list, answer_list)):
            if i == 0:
                msg += self.system.apply()
            question = self._inject_modal_tokens(question)
            msg += self.format_user.apply(content=question)
            msg += self.format_assistant.apply(content=answer)
        return msg
    
    def make_labels(self, input_ids, prompt, tokenizer):
        labels = copy.deepcopy(input_ids)
        sep, eos_token = self.separator.apply()
        total_len = int(labels.ne(tokenizer.pad_token_id).sum())
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            total_len += prompt.count(eos_token)
        rounds = prompt.split(eos_token)
        eos_token_length = len(tokenizer.encode(eos_token))
        labels, cur_len = self._make_masks(labels, tokenizer, sep, eos_token_length, rounds)
        if cur_len < tokenizer.model_max_length:
            import time
            if cur_len != total_len:
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
                print("number of rounds: ", len(rounds) - 1)
                print("rounds: ", rounds[:-1])
                print("prompt: ", prompt)
                print(labels)
                print(input_ids)
                time.sleep(5)
                labels[:] = IGNORE_INDEX
        return labels
        
        
        
    def _make_masks(self, labels, tokenizer, sep, eos_token_length, rounds):
        cur_len = 0
        for rou in rounds:
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(self.tokenizer_image_token(rou, tokenizer)) + eos_token_length
            instruction_len = len(self.tokenizer_image_token(parts[0], tokenizer)) - 1
            labels[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        labels[cur_len:] = IGNORE_INDEX
        return labels, cur_len
        
    @classmethod    
    def tokenizer_image_token(cls, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, sensor_token_index=SENSOR_TOKEN_INDEX, return_tensors=None):
        token_map = {
            DEFAULT_IMAGE_TOKEN: image_token_index,
            DEFAULT_SENSOR_TOKEN: sensor_token_index,
        }
        pattern_tokens = [re.escape(tok) for tok in token_map.keys()]
        pattern = f"({'|'.join(pattern_tokens)})" if pattern_tokens else None

        if pattern:
            prompt_chunks = re.split(pattern, prompt)
        else:
            prompt_chunks = [prompt]

        input_ids = []
        first_segment = True

        for chunk in prompt_chunks:
            if chunk == "" or chunk is None:
                continue
            if chunk in token_map:
                input_ids.append(token_map[chunk])
                first_segment = False
                continue
            tokenized_chunk = tokenizer(chunk).input_ids
            if not tokenized_chunk:
                continue
            if first_segment:
                first_segment = False
            else:
                if tokenized_chunk[0] == tokenizer.bos_token_id:
                    tokenized_chunk = tokenized_chunk[1:]
            input_ids.extend(tokenized_chunk)

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    def _inject_modal_tokens(self, question: str) -> str:
        token_formatters = {
            DEFAULT_IMAGE_TOKEN: getattr(self, 'format_image_token', None),
            DEFAULT_SENSOR_TOKEN: getattr(self, 'format_sensor_token', None),
        }
        pattern_tokens = [re.escape(tok) for tok, fmt in token_formatters.items() if fmt is not None]
        if not pattern_tokens:
            return question

        pattern = f"({'|'.join(pattern_tokens)})"
        parts = re.split(pattern, question)
        rebuilt = []
        for part in parts:
            if part == "" or part is None:
                continue
            formatter = token_formatters.get(part)
            if formatter is not None:
                rebuilt.append(formatter.apply(content=""))
            else:
                rebuilt.append(part)
        return ''.join(rebuilt)


