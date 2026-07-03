# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Inference-time data loading for LongBench.

`get_dataset`      — pulls the raw LongBench split.
`get_val_dataset`  — applies the model's chat template and tokenizes.
`MyCollator`       — left-pads to the batch max length so a single kv-cache
                     alignment works across the batch.
"""

from dataclasses import dataclass
from typing import Optional

from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

import eval.longbench_utils.eval_long_bench as longbench_eval
from eval.longbench_utils.constants import LONGBENCH_DATASET


# LongBench tasks that should be passed raw (no chat template).  These are
# instruction-following / classification-style tasks where the chat wrapper
# hurts accuracy.
_NO_CHAT_TEMPLATE_DATASETS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}


def get_dataset(tokenizer, pipeline_config, dataset_config, max_size=1_000_000_000):
    """Load a LongBench task and attach `prompt` / `answer` / `idx` fields."""
    ds_name = dataset_config["dataset"]
    if ds_name not in LONGBENCH_DATASET:
        raise ValueError(f"Unknown dataset: {ds_name}")

    ds = longbench_eval.load_data(ds_name)
    data = [{**dict(example), "idx": idx} for idx, example in enumerate(ds)]

    def tokenize_sample(sample):
        prompt = dataset_config["instruction"].format(**sample)
        answers = sample.get("answer", sample.get("answers", ""))
        answer = answers[0] if answers else ""
        return {"prompt": prompt, "answer": answer, "idx": sample["idx"]}

    keys = data[0].keys()
    dataset = Dataset.from_dict({k: [d[k] for d in data] for k in keys})
    return dataset.map(tokenize_sample, remove_columns=list(dataset.features), num_proc=32)


def get_val_dataset(base_dataset_valid, config, tokenizer):
    """Tokenize each sample's prompt, applying the chat template where appropriate."""
    dataset_name = config.get("eval_params", {}).get("dataset", "")
    use_chat_template = dataset_name not in _NO_CHAT_TEMPLATE_DATASETS
    max_len = config["pipeline_params"]["max_model_len"]

    def process_sample(sample):
        if use_chat_template:
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": sample["prompt"]},
            ]
            try:
                input_ids = tokenizer.apply_chat_template(
                    conversation, tokenize=True, add_generation_prompt=True, enable_thinking=False,
                )
            except TypeError:
                input_ids = tokenizer.apply_chat_template(
                    conversation, tokenize=True, add_generation_prompt=True,
                )
        else:
            input_ids = tokenizer(sample["prompt"], add_special_tokens=False)["input_ids"]

        # Middle-truncate long prompts to max_model_len.
        if len(input_ids) > max_len:
            input_ids = input_ids[: max_len // 2] + input_ids[-max_len // 2 :]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "prompt_len": len(input_ids),
            "idx": sample["idx"],
        }

    return base_dataset_valid.map(
        process_sample, remove_columns=list(base_dataset_valid.features), num_proc=32,
    )


@dataclass
class MyCollator:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features, return_tensors=None):
        # Left-pad so every sequence in the batch ends at the same position;
        # this maximizes KV-cache reuse across the batch.
        max_length = max(len(feature["input_ids"]) for feature in features)

        for feature in features:
            n_pad = max_length - len(feature["input_ids"])
            feature["prompt_len"] += n_pad
            feature["input_ids"] = [self.tokenizer.pad_token_id] * n_pad + feature["input_ids"]
            feature["attention_mask"] = [0] * n_pad + feature["attention_mask"]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer, features, padding=True, pad_to_multiple_of=None, return_tensors="pt",
        )
        return batch
