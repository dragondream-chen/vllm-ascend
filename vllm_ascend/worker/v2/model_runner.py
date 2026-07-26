# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

from contextlib import contextmanager

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.mm.lora import set_active_mm_loras
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.forward_context import BatchDescriptor
from vllm.v1.worker.gpu import model_runner as vllm_model_runner
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    get_uniform_token_count,
)
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    combine_sampled_and_draft_tokens,
    expand_idx_mapping,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner, ExecuteModelState
from vllm.v1.worker.gpu.lora_utils import (
    get_num_active_loras_for_dispatch,
)
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer


from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import (
    MoECommType,
    get_mc2_tokens_capacity,
    override_mrv2_in_profile_run,
    select_moe_comm_method,
    set_mc2_mask,
    set_mc2_tokens_capacity,
    set_ascend_forward_context,
)
from vllm_ascend.ops.rotary_embedding import set_cos_and_sin, update_cos_sin
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
from vllm_ascend.worker.v2.attn_utils import build_attn_state
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers
from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator
from vllm_ascend.worker.v2.states import AscendRequestState
from vllm_ascend.worker.v2.utils import torch_cuda_wrapper


class NPUModelRunner(GPUModelRunner):
    """Model runner for Ascend NPUs."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # Ascend-specific configurations
        self.ascend_config = get_ascend_config()
        # The following features are not yet supported in Ascend NPU model runner v2:
        # - Context parallelism (prefill or decode)
        # - Dynamic EPLB
        parallel_config = vllm_config.parallel_config
        if parallel_config.prefill_context_parallel_size > 1 or parallel_config.decode_context_parallel_size > 1:
            raise NotImplementedError("Context parallelism is not supported by Ascend NPU model runner v2.")

        if self.ascend_config.eplb_config.dynamic_eplb:
            raise NotImplementedError("dynamic_eplb is not supported by Ascend NPU model runner v2.")

        with torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        # because we will override these attribute, delete these attribute to
        # make sure it's collected by python gc immediately.
        del self.req_states
        del self.input_buffers
        del self.speculator

        # we define AscendEagleSpeculator in vllm_ascend.worker.v2.spec_decode.eagle.speculator
        # init_speculator will return AscendEagleSpeculator when eagle is used.
        # so here we just call init_speculator to reinitialize speculator.
        self.speculator: AscendEagleSpeculator | None = None
        if self.speculative_config is not None:
            self.speculator = init_speculator(self.vllm_config, self.device)

        # AscendRequestState has extra `num_computed_tokens_cpu` attribute.
        # so reinitialize req_states here.
        self.req_states: AscendRequestState = AscendRequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            num_speculative_steps=self.num_speculative_steps,
            vocab_size=self.vocab_size,
            device=self.device,
        )
        # AscendInputBuffers has extra `seq_lens_cpu` attribute.
        # so reinitialize input_buffers here.
        self.input_buffers: AscendInputBuffers = AscendInputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=self.device,
        )

        # we need to copy num_computed_tokens back to cpu to help
        # update actual seq_lens_cpu. gpu attention backend doesn't need these
        # attributes, cause their attention backends doesn't use seq_lens_cpu.
        # and seq_lens_cpu is deprecated in gpu_model_runner_v2.
        self.num_computed_tokens_event = torch.npu.Event()
        self.num_computed_tokens_stream = torch.npu.Stream()
        self.num_computed_tokens_cpu = torch.empty(
            self.max_num_reqs,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )

        # Set _mc2_tokens_capacity and _reserved_mc2_mask for MoE communication optimization.
        # TODO: remove set_cos_and_sin (together with update_cos_sin) when mla can properly handle cos/sin internally
        self.decode_query_len = self.num_speculative_steps + 1
        set_cos_and_sin(vllm_config, self.max_num_reqs, self.decode_query_len, self.dtype, self.device)
        set_mc2_tokens_capacity(vllm_config, self.max_num_reqs, self.decode_query_len)
        set_mc2_mask(vllm_config, self.device)

        # we need to update full graph params in run_fullgraph,
        # so create a stream to update full graph params.
        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.update_stream: torch.npu.Stream = torch.npu.Stream()

        # we need to use return value of `get_cudagraph_and_dp_padding`
        # to set forward_context in `run_fullgraph`.
        # so we can inherit `execute_model` method.
        self.cudagraph_and_dp_padding: tuple[int, torch.Tensor | None, int] | None = None

        # we need to use input_batch to set forward_context in run_fullgraph.
        # so we can inherit `execute_model` method.
        self.input_batch: AscendInputBatch | None = None

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        with graph_manager_wrapper(self):
            super().initialize_kv_cache(kv_cache_config)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        if not dummy_run:
            # Update the request states.
            self.update_pp_decode_requests()
            self.finish_requests(scheduler_output)
            self.free_states(scheduler_output)
            self.add_requests(scheduler_output)
            self.update_requests(scheduler_output)
            self.block_tables.apply_staged_writes()
            if scheduler_output.total_num_scheduled_tokens == 0:
                # No need to run the model.
                empty_output = self.kv_connector.no_forward(scheduler_output)
                return empty_output

        # Get batch descriptor and sync across DP ranks.
        num_reqs = len(scheduler_output.num_scheduled_tokens)
        num_toks = scheduler_output.total_num_scheduled_tokens
        max_query_len = max(scheduler_output.num_scheduled_tokens.values())
        uniform_tok_count = get_uniform_token_count(num_reqs, num_toks, max_query_len)

        num_active_loras = 0
        if self.lora_config:
            req_ids = list(scheduler_output.num_scheduled_tokens.keys())
            num_active_loras = get_num_active_loras_for_dispatch(
                self.lora_config, self.lora_state, req_ids, dummy_run
            )

        skip_compiled = False
        if self.is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
            # Encoder-decoder models such as Whisper should run eager/non-compiled
            # when encoder inputs are scheduled, because this step updates
            # cross-attention cache with dynamic encoder outputs.
            skip_compiled = True

        batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.cudagraph_manager,
            num_reqs,
            num_toks,
            uniform_tok_count,
            self.dp_size,
            self.dp_rank,
            need_eager=is_profile or skip_compiled,
            num_active_loras=num_active_loras,
        )

        if batch_desc.num_tokens == 0:
            # All DP ranks have zero tokens to run.
            empty_output = self.kv_connector.no_forward(scheduler_output)
            return empty_output

        if not dummy_run:
            # Common case.
            # Prepare all the inputs and copy to the input buffers.
            input_batch = self.prepare_inputs(scheduler_output, batch_desc)
            block_tables, slot_mappings = self.prepare_attn(input_batch)
            # Mamba "align" pre-copy: migrate recurrent state across block
            # boundaries before the forward. Runs only on real batches, and
            # before model_state.prepare_attn gathers num_accepted_tokens so the
            # boundary reset is visible to the attention metadata.
            self.model_state.preprocess_state(
                input_batch,
                block_tables,
                self.kv_cache_config,
                self.req_states.num_computed_tokens.gpu,
            )

            if self.lora_config:
                # Activate LoRA adapters.
                lora_inputs = self.lora_state.make_lora_inputs(
                    input_batch.req_ids,
                    input_batch.idx_mapping_np,
                    input_batch.num_scheduled_tokens,
                )
                self._set_active_loras(*lora_inputs)
        else:
            # No actual tokens to run. A dummy run for DP or memory profiling.
            input_batch = InputBatch.make_dummy(
                batch_desc.num_reqs or num_reqs,
                batch_desc.num_tokens,
                self.input_buffers,
            )
            if not skip_attn_for_dummy_run:
                block_tables, slot_mappings = self.prepare_dummy_attn(input_batch)
            else:
                assert batch_desc.cg_mode != CUDAGraphMode.FULL, (
                    "Attention metadata must be prepared for dummy runs when using "
                    "FULL cudagraph mode."
                )
                block_tables = None
                slot_mappings = None

        attn_metadata = None
        slot_mappings_by_layer = None
        if not (dummy_run and skip_attn_for_dummy_run):
            assert slot_mappings is not None
            slot_mappings_by_layer = build_slot_mappings_by_layer(
                slot_mappings, self.kv_cache_config
            )
            assert block_tables is not None
            attn_metadata = self.model_state.prepare_attn(
                input_batch,
                batch_desc.cg_mode,
                block_tables,
                slot_mappings,
                self.attn_groups,
                self.kv_cache_config,
            )

        input_ids = input_batch.input_ids
        inputs_embeds = None
        if self.supports_mm_inputs and self.is_first_pp_rank:
            # Run MM encoder (if needed) and get multimodal embeddings.
            # Only first PP rank prepares multimodal embeddings.
            if dummy_run:
                # Obtain mm embeddings of correct shape for compiled model.
                inputs_embeds = self.model_state.dummy_inputs_embeds(
                    input_batch.num_tokens_after_padding
                )
            else:
                scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
                if self.lora_config is not None:
                    set_active_mm_loras(
                        model=self.model,
                        lora_manager=self.lora_manager,
                        encoder_cache=self.encoder_cache,
                        req_id_to_index=self.req_states.req_id_to_index,
                        lora_state=self.lora_state,
                        scheduled_encoder_inputs=scheduled_encoder_inputs,
                    )
                inputs_embeds = self.model_state.get_mm_embeddings(
                    scheduled_encoder_inputs, input_batch, self.req_states
                )
            if inputs_embeds is not None and not self.model.requires_raw_input_tokens:
                input_ids = None

        model_inputs = {
            "input_ids": input_ids,
            "positions": input_batch.positions,
            "inputs_embeds": inputs_embeds,
            "intermediate_tensors": None,
            # NOTE: Values returned by `prepare_inputs` will override the default
            # values above.
            **self.model_state.prepare_inputs(input_batch, self.req_states),
        }
        if not self.is_first_pp_rank:
            # Update for non-first PP ranks.
            model_inputs["input_ids"] = None
            model_inputs["inputs_embeds"] = None

            # Prepare the intermediate tensors.
            assert intermediate_tensors is not None
            assert self.intermediate_tensors is not None
            n = input_batch.num_tokens_after_padding
            new_tensors = {
                k: v[:n]
                if dummy_run
                else v[:n].copy_(intermediate_tensors.tensors[k][:n])
                for k, v in self.intermediate_tensors.tensors.items()
            }
            model_inputs["intermediate_tensors"] = IntermediateTensors(new_tensors)
            del intermediate_tensors

        # Update the EPLB meta.
        self.eplb.prepare_forward(self.model_config, input_batch.num_tokens)

        # Run model.
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            # Use explicit cudagraph replay for FULL mode.
            # NOTE(woosuk): Here, we don't need to pass the input tensors,
            # because they are already copied to the CUDA graph input buffers.
            assert self.cudagraph_manager is not None
            self.kv_connector.pre_forward(scheduler_output)
            model_output = self.cudagraph_manager.run_fullgraph(batch_desc)
        else:
            # For piecewise and eager mode, just call model().
            batch_descriptor = BatchDescriptor(
                num_tokens=input_batch.num_tokens_after_padding,
                has_lora=self.lora_config is not None,
                num_active_loras=batch_desc.num_active_loras,
            )

            with set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=input_batch.num_tokens_after_padding,
                aclgraph_runtime_mode=batch_desc.cg_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=batch_descriptor,
                slot_mapping=slot_mappings_by_layer,
                skip_compiled=skip_compiled,
                is_padding=input_batch.is_padding,
                input_ids=input_ids,
            ):
                self.kv_connector.pre_forward(scheduler_output)
                if batch_desc.cg_mode == CUDAGraphMode.PIECEWISE:
                    # Run the PIECEWISE graph (compiled PW cudagraph or breakable
                    # cudagraph, chosen inside run_pw_graph). cg_mode is only
                    # PIECEWISE after the cudagraph manager exists.
                    assert self.cudagraph_manager is not None
                    model_output = self.cudagraph_manager.run_pw_graph(
                        self.model, model_inputs
                    )
                else:
                    # Eager (NONE): call the raw model directly.
                    model_output = self.model(**model_inputs)

        if self.is_last_pp_rank:
            if self.use_aux_hidden_state_outputs:
                assert isinstance(model_output, tuple)
                hidden_states, aux_hidden_states = model_output
            else:
                assert isinstance(model_output, torch.Tensor)
                hidden_states = model_output
                aux_hidden_states = None
            output_intermediate_tensors = None
        else:
            assert isinstance(model_output, IntermediateTensors)
            hidden_states = None
            aux_hidden_states = None
            output_intermediate_tensors = model_output

        finished_req_ids = scheduler_output.finished_req_ids
        self.execute_model_state = ExecuteModelState(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings_by_layer=slot_mappings_by_layer,
            hidden_states=hidden_states,
            aux_hidden_states=aux_hidden_states,
            finished_req_ids=finished_req_ids,
        )

        if not self.is_last_pp_rank:
            # Non-last PP rank: return IntermediateTensors for sending.
            return output_intermediate_tensors
        return None

    @torch.inference_mode()
    def profile_run(self) -> None:
        """Override GPUModelRunner.profile_run for Ascend NPUs.
        When running moe models, we need an extra dummy run with mc2_tokens_capacity tokens to reserve
        necessary HCCL buffer for the MC2 operator before standard `profile_run`. Additionally, we set
        override_mrv2_in_profile_run to True to force moe load to be balanced when executing `profile_run`
        """
        mc2_tokens_capacity = get_mc2_tokens_capacity()
        with override_mrv2_in_profile_run(True):
            if (
                mc2_tokens_capacity is not None
                and self.max_num_tokens > mc2_tokens_capacity
                and select_moe_comm_method(mc2_tokens_capacity, self.vllm_config)
                in {MoECommType.MC2, MoECommType.FUSED_MC2}
            ):
                self._dummy_run(mc2_tokens_capacity, skip_attn=True, is_profile=True)
            super().profile_run()

    def prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        batch_desc: BatchExecutionDescriptor,
    ) -> AscendInputBatch:
        """Override GPUModelRunner.prepare_inputs for Ascend NPUs.
        npu attention backends need seq_lens_cpu to work.
        so we need to prepare seq_lens_cpu here.
        """
        num_tokens = scheduler_output.total_num_scheduled_tokens
        num_tokens_after_padding = batch_desc.num_tokens
        assert num_tokens > 0
        num_tokens_per_req = scheduler_output.num_scheduled_tokens
        num_reqs = len(num_tokens_per_req)

        # Decode first, then prefill.
        # batch_idx -> req_id
        req_ids = sorted(num_tokens_per_req, key=num_tokens_per_req.get)  # type: ignore

        self._update_seq_lens_cpu(scheduler_output, req_ids)

        numtoks_iter = map(num_tokens_per_req.get, req_ids)
        num_scheduled_tokens = np.fromiter(numtoks_iter, dtype=np.int32, count=num_reqs)
        num_valid_tokens = num_scheduled_tokens
        if scheduler_output.scheduled_spec_decode_tokens:
            num_valid_tokens = np.array(
                [
                    num_tokens - len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
                    for num_tokens, i in zip(num_scheduled_tokens, req_ids)
                ],
                dtype=np.int32,
            )
        attn_state = build_attn_state(
            self.vllm_config,
            self.input_buffers.seq_lens_np,
            num_reqs,
            num_scheduled_tokens,
            num_valid_tokens,
        )
        idx_mapping_iter = map(self.req_states.req_id_to_index.get, req_ids)
        idx_mapping_np = np.fromiter(idx_mapping_iter, dtype=np.int32, count=num_reqs)
        idx_mapping_cpu = torch.from_numpy(idx_mapping_np)
        idx_mapping = async_copy_to_gpu(idx_mapping_cpu, device=self.device)

        # Get the number of draft tokens for each request.
        draft_tokens = scheduler_output.scheduled_spec_decode_tokens
        num_draft_tokens_per_req: np.ndarray | None = None
        if not draft_tokens:
            # No draft token scheduled (common case).
            total_num_draft_tokens = 0
            total_num_logits = num_reqs
            cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
            cu_num_logits = torch.arange(num_reqs + 1, device=self.device, dtype=torch.int32)
            expanded_idx_mapping = idx_mapping
            expanded_local_pos = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        else:
            num_draft_tokens_arr = np.array(
                [len(draft_tokens.get(req_id, ())) for req_id in req_ids],
                dtype=np.int32,
            )
            num_draft_tokens_per_req = num_draft_tokens_arr
            total_num_draft_tokens = int(num_draft_tokens_arr.sum())
            total_num_logits = num_reqs + total_num_draft_tokens

            num_logits = num_draft_tokens_arr + 1
            cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
            cu_num_logits_np[0] = 0
            np.cumsum(num_logits, out=cu_num_logits_np[1:])
            cu_num_logits = async_copy_to_gpu(cu_num_logits_np, device=self.device)

            max_expand_len = self.num_speculative_steps + 1
            expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
                idx_mapping, total_num_logits, cu_num_logits, max_expand_len
            )

        # Get query_start_loc.
        # NOTE: For FULL mode we change +1 to +2 to reserve extra space for padding.
        # See _pad_query_start_loc_for_fia.
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        query_start_loc_np = np.empty(self.max_num_reqs + 2, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1 : num_reqs + 1])
        # Pad for full CUDA graph mode.
        # Some attention backends like FA3 require query_start_loc to be non-decreasing.
        query_start_loc_np[num_reqs + 1 :] = num_tokens

        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            # This is only required for vllm-ascend.
            query_start_loc_np, num_reqs_padded = self._pad_query_start_loc_for_fia(
                num_tokens_after_padding,
                num_reqs_padded,
                num_reqs,
                query_start_loc_np,
                batch_desc.cg_mode,
                batch_desc.num_reqs,
            )

        async_copy_to_gpu(query_start_loc_np, out=self.input_buffers.query_start_loc)

        query_start_loc_np = query_start_loc_np[: num_reqs_padded + 1]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        prefill_len_np = self.req_states.prefill_len.np[idx_mapping_np]
        num_computed_prefill_tokens_np = self.req_states.num_computed_prefill_tokens[idx_mapping_np]
        is_prefilling_np = num_computed_prefill_tokens_np < prefill_len_np

        # Get prefill tokens if any.
        if np.any(is_prefilling_np):
            prepare_prefill_inputs(
                self.input_buffers.input_ids,
                self.req_states.next_prefill_tokens,
                idx_mapping,
                query_start_loc,
                self.req_states.all_token_ids.gpu,
                self.req_states.prefill_len.gpu,
                self.req_states.num_computed_tokens.gpu,
            )

        # Prepare positions and seq_lens.
        prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            self.req_states.num_computed_tokens.gpu,
            self.input_buffers.positions,
            self.input_buffers.seq_lens,
        )
        seq_lens = self.input_buffers.seq_lens[:num_reqs]

        # Pad for full CUDA graph mode.
        self.input_buffers.seq_lens_np[num_reqs_padded:] = 0

        # Some input token ids are directly read from the last sampled tokens
        # and draft tokens. Also, get the logits indices to sample tokens from.
        logits_indices = combine_sampled_and_draft_tokens(
            self.input_buffers.input_ids,
            idx_mapping,
            self.req_states.last_sampled_tokens,
            query_start_loc,
            seq_lens,
            self.req_states.prefill_len.gpu,
            self.req_states.draft_tokens,
            cu_num_logits,
            total_num_logits,
        )

        input_ids = self.input_buffers.input_ids[:num_tokens_after_padding]
        positions = self.input_buffers.positions[:num_tokens_after_padding]

        # CPU upper bound on seq_lens (num_computed_tokens + num_scheduled_tokens).
        # Added by vLLM PR #40654 to avoid GPU->CPU sync for seq_lens.
        seq_lens_cpu_upper_bound_np = np.zeros(num_reqs_padded, dtype=np.int32)
        np.add(
            self.req_states.num_computed_tokens_np[idx_mapping_np],
            num_scheduled_tokens,
            out=seq_lens_cpu_upper_bound_np[:num_reqs],
        )
        seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens_cpu_upper_bound_np)
        num_computed_tokens_np = self.req_states.num_computed_tokens_np[idx_mapping_np]
        max_seq_len_np = None
        if getattr(self, "use_pp", False):
            # max_seq_len is only consumed by the PP `compute_need_sampled_mask`.
            max_seq_len_np = self.req_states.max_seq_len[idx_mapping_np]

        self.input_batch = AscendInputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs_padded,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
            num_draft_tokens=total_num_draft_tokens,
            num_draft_tokens_per_req=num_draft_tokens_per_req,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=None,  # TODO(Ronald1995): support cp.
            is_prefilling_np=is_prefilling_np,
            num_computed_tokens_np=num_computed_tokens_np,
            prefill_len_np=prefill_len_np,
            num_computed_prefill_tokens_np=num_computed_prefill_tokens_np,
            max_seq_len_np=max_seq_len_np,
            input_ids=input_ids,
            positions=positions,
            is_padding=self.input_buffers.is_padding[:num_tokens_after_padding],
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=scheduler_output.has_structured_output_requests,
            # TODO: only populated for R-SWA (not supported yet).
            prompt_lens=None,
            # extra attributes for ascend npus.
            seq_lens_np=self.input_buffers.seq_lens_np,
            attn_state=attn_state,
        )

        # For mla/sfa, update cos/sin. Here is for execute_model.
        update_cos_sin(self.input_batch.positions)

        return self.input_batch

    def postprocess(
        self,
        input_batch,
        sampled_tokens,
        num_sampled,
        num_rejected,
    ):
        """Override GPUModelRunner.postprocess for Ascend NPUs.
        npu attention backends need seq_lens_cpu to work.
        so we need to copy num_computed_tokens back to cpu here.
        """
        super().postprocess(
            input_batch,
            sampled_tokens,
            num_sampled,
            num_rejected,
        )

        self._copy_num_computed_tokens_to_cpu()

    def postprocess_sampled(
        self,
        idx_mapping,
        sampled_tokens,
        num_sampled,
        num_rejected,
        query_start_loc=None,
    ):
        """Override GPUModelRunner.postprocess_sampled for Ascend NPUs."""
        super().postprocess_sampled(
            idx_mapping,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
        )

        self._copy_num_computed_tokens_to_cpu()

    def _copy_num_computed_tokens_to_cpu(self):
        # npu attention backend still need to use seq_lens_cpu,
        # we need to copy num_computed_tokens back to cpu.
        default_stream = torch.cuda.current_stream()
        assert self.num_computed_tokens_stream is not None
        assert self.num_computed_tokens_cpu is not None
        with torch.npu.stream(self.num_computed_tokens_stream):
            self.num_computed_tokens_stream.wait_stream(default_stream)
            self.num_computed_tokens_cpu.copy_(
                self.req_states.num_computed_tokens.gpu,
                non_blocking=True,
            )
            self.num_computed_tokens_event.record()

    def _update_seq_lens_cpu(
        self,
        scheduler_output: SchedulerOutput,
        req_ids: list[str],
    ):
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        # wait for num_computed_tokens copy to cpu stream to finish.
        self.num_computed_tokens_event.synchronize()
        for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
            req_index = self.req_states.req_id_to_index[req_id]
            # num_computed_tokens_cpu has reverted by num_rejected_tokens already.
            # in super postprocess method.
            self.req_states.num_computed_tokens_cpu[req_index] = self.num_computed_tokens_cpu[req_index]

        # update seq_lens_cpu
        for i, req_id in enumerate(req_ids):  # type: ignore
            req_index = self.req_states.req_id_to_index[req_id]
            num_computed_tokens = self.req_states.num_computed_tokens_cpu[req_index]
            self.input_buffers.seq_lens_cpu[i] = num_computed_tokens + num_scheduled_tokens[req_id]

    def eplb_warmup(self):
        # TODO(Ronald1995): just define the method in case calling error in
        # worker, implement it in the future.
        pass

    def _pad_query_start_loc_for_fia(
        self,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_reqs: int,
        query_start_loc_np: np.ndarray,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        batch_desc_num_reqs: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        This function is only designed to satisfied the constraint that when the layout is TND,
        the first dimension of `hidden_states` must equal the last element of `actual_seq_lengths_q`.
        """
        # TODO: need refactor later, related to vllm PR #34043 this pr delete func
        # relax_for_mixed_batch_cudagraphs, num_reqs no longer equals the actual number of requests.
        if cudagraph_runtime_mode == CUDAGraphMode.FULL:
            num_reqs_padded = num_reqs
        else:
            num_reqs_padded = batch_desc_num_reqs if batch_desc_num_reqs is not None else num_reqs

        if num_tokens_padded == num_reqs_padded * self.decode_query_len:
            # Uniform-batch case: num_reqs must be no greater than num_reqs_padded
            assert num_reqs <= num_reqs_padded

            last_loc = query_start_loc_np[num_reqs]
            query_start_loc_np[num_reqs + 1 : num_reqs_padded + 1] = (
                np.arange(1, num_reqs_padded + 1 - num_reqs) * self.decode_query_len + last_loc
            )
        else:
            # Mixed-batch case: num_reqs must equal num_reqs_padded
            assert num_reqs == num_reqs_padded

            # Insert a dummy request instead of setting query_start_loc[num_reqs] = num_tokens_padded directly
            query_start_loc_np[num_reqs_padded + 1] = num_tokens_padded
            num_reqs_padded = num_reqs_padded + 1

        return query_start_loc_np, num_reqs_padded


@contextmanager
def graph_manager_wrapper(model_runner):
    """Context manager to override graph manager."""
    original_graph_manager = vllm_model_runner.ModelCudaGraphManager

    def factory(
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        lora_capture_cases: list[int] | None = None,
    ):
        return ModelAclGraphManager(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
            model_runner,
            lora_capture_cases=lora_capture_cases,
        )

    try:
        vllm_model_runner.ModelCudaGraphManager = factory
        yield
    finally:
        vllm_model_runner.ModelCudaGraphManager = original_graph_manager
