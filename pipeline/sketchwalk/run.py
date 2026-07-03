import os
import re
import sys

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

import eval.longbench_utils.eval_long_bench as longbench_eval
from eval.longbench_utils.constants import LONGBENCH_DATASET

from pipeline.sketchwalk.CE import CE
from pipeline.sketchwalk.dataset import MyCollator, get_dataset, get_val_dataset
from pipeline.sketchwalk.modeling.modeling_llama import LlamaForCausalLM
from pipeline.sketchwalk.modeling.modeling_qwen3 import Qwen3ForCausalLM
from pipeline.sketchwalk.utils import set_seed


_LLAMA_MODELS = {
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
}
_QWEN3_MODELS = {
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-0.6B",
}


_SKETCHWALK_KEYS = (
    "random_walk_hadamard_dim",
    "random_walk_window",
    "random_walk_sink",
    "random_walk_kblocks_frac",
    "random_walk_degree",
    "random_walk_block_size",
    "random_walk_exact",
    "random_walk_query_window",
    "walk_damping",
)


def _apply_sketchwalk_config(cfg, pipeline_params):
    """Copy SketchWalk knobs from pipeline_params onto a HF model config."""
    for key in _SKETCHWALK_KEYS:
        if key in pipeline_params:
            setattr(cfg, key, pipeline_params[key])
    return cfg


def _build_model(model_name, pipeline_params):
    if model_name in _LLAMA_MODELS:
        cfg = _apply_sketchwalk_config(AutoConfig.from_pretrained(model_name), pipeline_params)
        model = LlamaForCausalLM.from_pretrained(model_name, config=cfg, torch_dtype=torch.bfloat16)
    elif model_name in _QWEN3_MODELS:
        cfg = _apply_sketchwalk_config(AutoConfig.from_pretrained(model_name), pipeline_params)
        model = Qwen3ForCausalLM.from_pretrained(model_name, config=cfg, torch_dtype=torch.bfloat16)
    else:
        raise ValueError(
            f"SketchWalk not implemented for {model_name!r}. "
            f"Supported: {sorted(_LLAMA_MODELS | _QWEN3_MODELS)}"
        )
    model.set_sparsity_mode(pipeline_params["sparsity_mode"])
    return model


def run(configs, args, logger):
    eval_params = configs["eval_params"]
    pipeline_params = configs["pipeline_params"]

    dataset_name = eval_params["dataset"]
    if dataset_name not in LONGBENCH_DATASET:
        raise ValueError(f"Only LongBench tasks are supported; got {dataset_name!r}.")

    ds_raw = longbench_eval.load_data(dataset_name)
    ground_truth = [ex["answers"] for ex in ds_raw]
    all_classes = ds_raw[0]["all_classes"]

    print("Config:", configs)
    set_seed(getattr(args, "seed", pipeline_params.get("seed", 41)))

    # ---- Model ----
    model = _build_model(pipeline_params["model_name"], pipeline_params)
    tokenizer = AutoTokenizer.from_pretrained(pipeline_params["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    model_module = CE(model, tokenizer)
    model_module.to("cuda")
    model_module.eval()
    device = next(model_module.parameters()).device

    # ---- Data ----
    base_dataset = get_dataset(tokenizer, pipeline_params, eval_params)
    valid_dataset = get_val_dataset(base_dataset, configs, tokenizer)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        num_workers=1,
        pin_memory=True,
        batch_size=1,
        collate_fn=MyCollator(tokenizer),
    )

    # ---- Generation eval ----
    max_new_tokens = eval_params["max_new_tokens"]
    pbar = tqdm(colour="blue", desc="Test Accuracy", total=len(valid_loader), dynamic_ncols=True)
    score = torch.tensor(0.0, device=device)
    total = torch.tensor(0.0, device=device)
    pred_map = {}

    with torch.no_grad():
        for idx, batch in enumerate(valid_loader):
            test_idx = int(batch["idx"][0])
            batch = {k: v.to(device) for k, v in batch.items() if v is not None and k != "idx"}
            assert len(batch["input_ids"]) == 1
            total += 1

            answer = model_module.generate(**batch, max_new_tokens=max_new_tokens)
            answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
            pred_map[test_idx] = answer
            score += longbench_eval.scorer(
                dataset_name, [answer], [ground_truth[idx]], all_classes
            )

            if idx < 50:
                print(f"Question {test_idx}: Answer = '{answer}'")
            pbar.update(1)
            pbar.set_description(f"Test accuracy: {round((score / total).item(), 2)}")
    pbar.close()

    max_idx = max(pred_map) if pred_map else -1
    all_predictions = [pred_map.get(i) for i in range(max_idx + 1)]

    final_score = (100 * score / total).item()
    print(f"Accuracy on validation set: {round(final_score, 2)}")
    sys.stdout.flush()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    processed = {"dataset": dataset_name, "score": final_score, "outputs": all_predictions}
    raw = {"total_score": score.item(), "total": total.item()}
    return processed, raw
