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
"""DRAFT, DO NOT MERGE: dumps for the R3 multi-turn boundary repro.

No-op unless ``NRL_R3_BOUNDARY_DEBUG_DIR`` is set. ``dump_call`` records what
vLLM computed for one chat call (every path); ``dump_row`` records the routes
a training row ends up with, plus the positions of its non-final turns' last
generated tokens. ``tools/r3_boundary_repro/check.py`` compares the two.
"""

import os
import uuid

import torch


def _save(kind: str, record: dict) -> None:
    root = os.environ.get("NRL_R3_BOUNDARY_DEBUG_DIR")
    if not root:
        return
    os.makedirs(os.path.join(root, kind), exist_ok=True)
    torch.save(record, os.path.join(root, kind, f"{uuid.uuid4().hex}.pt"))


def dump_call(prompt_token_ids, generation_token_ids, routes: torch.Tensor) -> None:
    """One vLLM chat call: full ``[prompt + generation, layers, topk]`` routes."""
    _save(
        "call",
        {
            "token_ids": [int(t) for t in prompt_token_ids]
            + [int(t) for t in generation_token_ids],
            "prompt_len": len(prompt_token_ids),
            "routes": routes.detach().to("cpu", torch.int16),
        },
    )


def dump_row(source: str, token_ids, routes: torch.Tensor, boundaries) -> None:
    """One training row's assembled routes and its turn-boundary positions."""
    _save(
        "row",
        {
            "source": source,
            "token_ids": [int(t) for t in token_ids],
            "routes": routes.detach().to("cpu", torch.int16),
            "boundaries": [int(b) for b in boundaries],
        },
    )
