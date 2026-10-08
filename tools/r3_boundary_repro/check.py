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
"""DRAFT: do training rows carry vLLM's real route at multi-turn boundaries?

Inputs (written by nemo_rl/experience/_r3_boundary_debug.py):
  call/*.pt  one per vLLM chat call: its full token ids, prompt length, and
             the routes vLLM computed for every position.
  row/*.pt   one per training row: its token ids, the routes it will train
             with, and its boundary positions.

A boundary is the last generated token of a non-final turn. The call that
generated it never ran it through the model, so that call has no real route
for it. The next call does: the boundary token is part of its prompt.

For every boundary we compare the row's route with the next call's route.

Exit code: 0 = all boundaries match, 1 = some don't, 2 = no boundaries found.
Usage: python check.py $NRL_R3_BOUNDARY_DEBUG_DIR
"""

import glob
import sys

import torch


def load_dumps(dump_dir: str, kind: str) -> list[dict]:
    return [torch.load(path) for path in glob.glob(f"{dump_dir}/{kind}/*.pt")]


def find_next_call(calls: list[dict], row_token_ids: list[int], boundary: int) -> dict:
    """The first call of this conversation whose prompt contains the boundary token."""
    candidates = []
    for call in calls:
        call_token_ids = call["token_ids"]
        is_same_conversation = call_token_ids == row_token_ids[: len(call_token_ids)]
        prompt_contains_boundary = call["prompt_len"] > boundary
        if is_same_conversation and prompt_contains_boundary:
            candidates.append(call)
    assert candidates, f"no dumped call has boundary position {boundary} in its prompt"
    return min(candidates, key=lambda call: call["prompt_len"])


def main(dump_dir: str) -> int:
    calls = load_dumps(dump_dir, "call")
    rows = load_dumps(dump_dir, "row")
    sources = sorted({row["source"] for row in rows})
    print(f"loaded {len(calls)} calls and {len(rows)} rows (paths: {sources})")

    num_match = 0
    num_mismatch = 0
    for row in rows:
        for boundary in row["boundaries"]:
            next_call = find_next_call(calls, row["token_ids"], boundary)
            real_route = next_call["routes"][boundary]
            training_route = row["routes"][boundary]

            if torch.equal(training_route, real_route):
                num_match += 1
            else:
                num_mismatch += 1
                if num_mismatch == 1:
                    print(f"first mismatch at position {boundary}:")
                    print(f"  training route: {training_route.tolist()}")
                    print(f"  real route:     {real_route.tolist()}")

    print(f"boundaries: {num_match} match, {num_mismatch} mismatch")
    if num_mismatch > 0:
        return 1
    if num_match == 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
