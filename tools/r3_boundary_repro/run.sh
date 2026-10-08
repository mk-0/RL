#!/bin/bash
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

# DRAFT: end-to-end check of R3 routes at multi-turn boundaries.
# One GRPO step, SC + vLLM (non-colocated) + Megatron + router replay, random
# tiny Qwen3-MoE, Gym's scripted multi-turn env (2-3 calls/rollout regardless
# of model output). Needs 2 GPUs on one node. Extra args are config overrides:
#   bash run.sh                              # token capture  -> expect BUG (exit 1)
#   bash run.sh ++token_capture.enabled=false  # token echo   -> expect OK  (exit 0)
# Exit code comes from check.py: 1 = bug, 0 = real routes, 2 = inconclusive.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
WORK=${R3_REPRO_WORK:-/tmp/r3_boundary_repro}
MODEL_DIR=$WORK/tiny_qwen3_moe
DATA=$PROJECT_ROOT/3rdparty/Gym-workspace/Gym/resources_servers/example_multi_turn_gymnasium/data/example.jsonl
export NRL_R3_BOUNDARY_DEBUG_DIR=$WORK/dumps
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
cd "$PROJECT_ROOT"

# dtensor_cfg was renamed to automodel_cfg on main (#4282); support both.
if grep -q "^  automodel_cfg:" examples/configs/grpo_math_1B.yaml; then
    TORCH_POLICY_CFG=automodel_cfg
else
    TORCH_POLICY_CFG=dtensor_cfg
fi

CONFIG=examples/nemo_gym/grpo_qwen3_30ba3b_instruct.yaml
# ``++`` for keys a given commit's YAML may not declare (read with .get).
OVERRIDES=(
    policy.model_name="$MODEL_DIR"
    policy.$TORCH_POLICY_CFG.enabled=false
    policy.megatron_cfg.enabled=true
    policy.megatron_cfg.tensor_model_parallel_size=1
    policy.megatron_cfg.pipeline_model_parallel_size=1
    policy.megatron_cfg.expert_model_parallel_size=1
    policy.megatron_cfg.context_parallel_size=1
    policy.megatron_cfg.sequence_parallel=false
    ++policy.router_replay.enabled=true
    ++token_capture.enabled=true
    policy.max_total_sequence_length=1024
    policy.generation.max_new_tokens=32
    policy.generation.vllm_cfg.tensor_parallel_size=1
    policy.generation.vllm_cfg.async_engine=true
    ++policy.generation.vllm_cfg.enable_prefix_caching=false
    ++policy.generation.vllm_kwargs.enable_chunked_prefill=false
    policy.generation.colocated.enabled=false
    policy.generation.colocated.resources.num_nodes=1
    policy.generation.colocated.resources.gpus_per_node=1
    cluster.gpus_per_node=2
    "env.nemo_gym.config_paths=[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,resources_servers/example_multi_turn_gymnasium/configs/example_multi_turn_gymnasium.yaml]"
    data.train.data_path="$DATA"
    data.validation.data_path="$DATA"
    grpo.num_prompts_per_step=4
    grpo.num_generations_per_prompt=2
    grpo.max_num_steps=1
    grpo.val_period=-1
    grpo.val_at_start=false
    grpo.async_grpo=null
    policy.train_global_batch_size=8
    policy.train_micro_batch_size=1
    checkpointing.enabled=false
    logger.wandb_enabled=false
    logger.tensorboard_enabled=false
    logger.log_dir="$WORK/logs"
    ++data_plane.enabled=true
    ++data_plane.impl=transfer_queue
    ++data_plane.backend=simple
    ++data_plane.simple.storage_capacity=1000000
    ++data_plane.simple.num_storage_units=2
    ++data_plane.claim_meta_poll_interval_s=0.5
    ++async_rl.sampler.name=in_order
    ++async_rl.sampler.max_lookahead_versions=0
    ++async_rl.min_groups_for_streaming_train=4
    ++async_rl.max_inflight_prompts=4
    ++async_rl.max_buffered_rollouts=4
    "$@"
)

# R3_REPRO_PRINT_OVERRIDES=1: print config + overrides (one per line) and exit.
if [[ -n "${R3_REPRO_PRINT_OVERRIDES:-}" ]]; then
    echo "$CONFIG"
    printf '%s\n' "${OVERRIDES[@]}"
    exit 0
fi

rm -rf "$NRL_R3_BOUNDARY_DEBUG_DIR" "$WORK/logs"
mkdir -p "$WORK/logs"

[[ -f $MODEL_DIR/config.json ]] || uv run --no-sync python "$SCRIPT_DIR/make_tiny_moe.py" "$MODEL_DIR"

# The run's own success is irrelevant: the dumps are written at finalize time,
# before training. Keep going so check.py always runs.
uv run --no-sync python examples/run_grpo_single_controller.py --config "$CONFIG" "${OVERRIDES[@]}" \
    2>&1 | tee "$WORK/run.log" || echo "[r3-repro] training run exited non-zero; checking dumps anyway"

uv run --no-sync python "$SCRIPT_DIR/check.py" "$NRL_R3_BOUNDARY_DEBUG_DIR"
