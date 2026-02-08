# train_grpo.py
from datasets import load_dataset
import torch
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward

from interventions_rl.model import qwen3, llama, interventions_utils
from interventions_rl.model.load_model import load_interventions_model

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

report, model = load_interventions_model(
    hf_model_name_or_path="Qwen/Qwen3-1.7B",
    model_class=qwen3.Qwen3ForCausalLM,
    ic_config=interventions_utils.InterventionsConfig(
        intervention_type="LoreftIntervention",
        intervention_layers="all",
        low_rank_dimension=128,
        dropout=0.0,
        act_fn="gelu",
        init_orth=True,
    ),
    map_dtype=torch.float32,
    map_device=torch.device("cuda"),
    trust_remote_code=True,
)

print(report.summary())

config = GRPOConfig(
    use_vllm=True,
    vllm_mode="colocate",
)
trainer = GRPOTrainer(
    model=model, reward_funcs=accuracy_reward, train_dataset=dataset, args=config
)
trainer.train()
