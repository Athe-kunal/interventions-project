"""
InterventionsGRPOTrainer: GRPOTrainer subclass that supports ReFT interventions with vLLM.

LoRA can be merged into base weights before syncing to vLLM (merge -> sync -> unmerge).
ReFT interventions CANNOT be merged -- they apply h + R^T(sigma(Wh + b) - Rh) to the
residual stream after each decoder block, which is not absorbable into any single weight.

Strategy: since base weights are frozen (only intervention params are trainable), vLLM
always has the correct base weights. We skip intervention params during weight sync.
Generation uses the base model via vLLM; training uses base + interventions for loss.
"""

from contextlib import nullcontext

from trl import GRPOTrainer


class InterventionsGRPOTrainer(GRPOTrainer):
    """GRPOTrainer that skips intervention parameters during vLLM weight sync.

    ReFT interventions add extra modules (learned_source, rotate_layer) to each
    decoder layer. These don't exist in vLLM's model and can't be merged into
    base weights. Since base weights are frozen, vLLM always has the right weights
    and we only need to skip intervention params during sync.
    """

    def _move_model_to_vllm(self):
        # If this is also a PEFT model, delegate to the parent (it handles merge/unmerge)
        try:
            from peft import PeftModel
            if isinstance(self.model, PeftModel):
                return super()._move_model_to_vllm()
        except ImportError:
            pass

        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed
            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if self.is_fsdp_enabled:
            # FSDP path: use parent's FSDP sync but filter intervention params
            fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
            fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
            if fsdp_version == 1:
                self._sync_fsdp1_params_to_vllm(self.model)
            elif fsdp_version == 2:
                self._sync_fsdp2_params_to_vllm(self.model)
        else:
            for name, param in self.model.named_parameters():
                # Skip intervention params -- they don't exist in vLLM's model
                # and can't be merged into base weights (unlike LoRA).
                if "intervention" in name:
                    continue

                name = self._fix_param_name_to_vllm(name)
                with gather_if_zero3([param]):
                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(name, param.data)])

        # Reset KV cache
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()
