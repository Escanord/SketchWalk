# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Thin inference wrapper around a causal LM.  Provides a greedy `.generate()`
that returns the decoded string of the newly generated tokens.
"""
import torch
import torch.nn as nn

from pipeline.sketchwalk.utils import _strip_special_tokens


class CE(nn.Module):
    def __init__(self, base_causallm, tokenizer):
        super().__init__()
        self.base_causallm = base_causallm
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.embedding = self.base_causallm.get_input_embeddings()

    def eval(self):
        self.base_causallm.config.use_cache = True
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,
        prompt_len,
        max_new_tokens=16,
        **kwargs,
    ):
        assert input_ids.shape[0] == 1, "only batch_size == 1 is supported"

        tokens = input_ids[0].detach().tolist()
        position_ids = torch.arange(
            0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
        ).reshape(1, -1)
        inputs_embeds = self.embedding(input_ids)
        kv_cache = None
        compute_range = (0, input_ids.shape[1])

        for _ in range(max_new_tokens):
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds[:, compute_range[0] : compute_range[1], :],
                attention_mask=attention_mask,
                position_ids=position_ids[:, compute_range[0] : compute_range[1]],
                past_key_values=kv_cache,
            )
            kv_cache = outputs.past_key_values

            next_token = torch.argmax(outputs.logits[0, -1]).item()
            tokens.append(next_token)
            if next_token == self.eos_token_id:
                break

            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

            new_attention = torch.ones(
                (attention_mask.size(0), 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([attention_mask, new_attention], dim=1)

            new_pos = position_ids[:, -1:] + 1
            position_ids = torch.cat([position_ids, new_pos], dim=1)
            compute_range = (compute_range[1], compute_range[1] + 1)

        answer = tokens[prompt_len:]
        output = self.tokenizer.decode(answer, spaces_between_special_tokens=False)
        output = _strip_special_tokens(output, self.tokenizer)
        return output
