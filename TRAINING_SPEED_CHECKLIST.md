# GRPO Training Speed Checklist

Use this checklist when `scripts/train_grpo.sh` feels slow.

## 1) First: identify your bottleneck (5-minute sanity check)

- [ ] Run `nvidia-smi dmon -s pucm` during training.
  - If **GPU util is low** (<40-50%), you are likely CPU/data/gen-bound.
  - If **memory is near full** and steps are unstable, reduce sequence/generation load.
- [ ] Watch training logs and note **seconds/step** before and after each change.
- [ ] Confirm all expected GPUs are visible (`CUDA_VISIBLE_DEVICES=0,1,2,3` in script).

## 2) Biggest speed levers in this repo (change these first)

Your current script settings are expensive for rollout:
- `max_completion_length=4096`
- `max_prompt_length=512`
- `num_generations=8`
- `per_device_train_batch_size=8`

These together create very large token throughput requirements.

### Suggested fast-debug profile

- [ ] Set `--config.training.max_completion_length 1024` (or 512).
- [ ] Set `--config.training.max_prompt_length 256`.
- [ ] Set `--config.training.num_generations 4` (or even 2 for debugging).
- [ ] Keep `--config.training.per_device_train_batch_size` high enough to saturate GPU, but reduce if OOM/slowdowns happen.

## 3) vLLM-specific optimization checklist

In `scripts/train_grpo.sh`, default is `VLLM_MODE=colocate` and `VLLM_GPU_MEMORY_UTILIZATION=0.1`.

- [ ] Increase `VLLM_GPU_MEMORY_UTILIZATION` from `0.1` to `0.3-0.6` (test gradually).
- [ ] Keep tensor parallel aligned with GPU count (`VLLM_TENSOR_PARALLEL_SIZE=4` on 4 GPUs).
- [ ] Try `VLLM_MODE=server` if colocate mode causes contention with training memory.
- [ ] Leave `VLLM_SKIP_WARMUP=1` enabled (already set) for faster start.

## 4) Model/intervention complexity checklist

- [ ] Use a smaller base model for experiments (e.g., 1.5B before 3B+).
- [ ] Reduce intervention cost:
  - Try `--config.interventions.intervention_layers last_only` or `odd_only`.
  - Reduce `--config.interventions.low_rank_dimension` (e.g., 128 -> 64).
- [ ] Keep `dtype=bfloat16` (already default in config).

## 5) Training loop overhead checklist

- [ ] Increase `--config.training.logging_steps` (e.g., 1 -> 10/20).
- [ ] Increase `--config.training.save_steps` (e.g., 64 -> 256+), especially for short experiments.
- [ ] Keep `--config.training.report_to none` when speed testing (already set in script).
- [ ] If compatible in your setup, try enabling `--config.training.use_liger_kernel true` and benchmark.

## 6) Data pipeline checklist

- [ ] Start with fewer examples for tuning runs:
  - set `--config.dataset.example_numbers` to a realistic debug size (e.g., 5k/10k), not full dataset.
- [ ] Ensure dataset is cached locally (avoid repeated remote fetch penalties).

## 7) Distributed/system checklist

- [ ] Confirm your `accelerate` config matches your hardware (`scripts/accelerate/ds_zero2_4gpu.yaml`).
- [ ] Use `torchrun`/`accelerate` on NVLink-connected GPUs when possible.
- [ ] Avoid running heavy jobs on the same machine while training.
- [ ] Pin CPU threads if dataloader becomes bottleneck (advanced tuning).

## 8) Practical tuning order (recommended)

1. Reduce generation lengths (`max_completion_length`, `max_prompt_length`).
2. Reduce `num_generations`.
3. Increase vLLM memory utilization from `0.1`.
4. Reduce intervention layers/rank.
5. Adjust batch size to balance utilization vs memory.
6. Reduce logging/checkpoint frequency.

## 9) Example faster command overrides

```bash
./scripts/train_grpo.sh \
  --config.training.max_completion_length 1024 \
  --config.training.max_prompt_length 256 \
  --config.training.num_generations 4 \
  --config.training.logging_steps 10 \
  --config.training.save_steps 256
```

If you keep the shell script unchanged, set env vars before running:

```bash
export VLLM_GPU_MEMORY_UTILIZATION=0.4
export VLLM_MODE=colocate
./scripts/train_grpo.sh
```

## 10) What to record for each experiment

- [ ] tokens/sec (or samples/sec)
- [ ] sec/step
- [ ] GPU util + memory util
- [ ] reward trend (to ensure speed improvements do not destroy learning)

A speedup is only useful if reward/quality does not collapse.
