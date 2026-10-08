# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DRAFT: random-weight tiny Qwen3-MoE (16 experts, top-2) + Qwen3 tokenizer.

16 experts / top-2 keeps a real route distinguishable from the dummy
``arange(top_k)``; the scripted multi-turn env makes output quality irrelevant.
Usage: python make_tiny_moe.py <out_dir>
"""

import sys

import torch
from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM

out = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
config = Qwen3MoeConfig(
    vocab_size=len(tokenizer),
    hidden_size=512,
    intermediate_size=1024,
    moe_intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=64,
    num_experts=16,
    num_experts_per_tok=2,
    norm_topk_prob=True,
    decoder_sparse_step=1,
    mlp_only_layers=[],
    max_position_embeddings=4096,
    tie_word_embeddings=False,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
    torch_dtype="bfloat16",
)
torch.manual_seed(0)
Qwen3MoeForCausalLM(config).to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
tokenizer.save_pretrained(out)
print(f"saved tiny Qwen3-MoE to {out}")
