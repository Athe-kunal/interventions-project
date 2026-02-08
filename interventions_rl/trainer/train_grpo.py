import torch
from trl import GRPOTrainer, GRPOConfig
from loguru import logger
from interventions_rl.model import qwen3, llama, interventions_utils
from interventions_rl.model.load_model import load_interventions_model
from interventions_rl.data import open_r1

dataset = open_r1.load_openr1_dataset(
    "open-r1/DAPO-Math-17k-Processed", example_numbers=100, test_split_ratio=0.1
)

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

logger.info(report.summary())

config = GRPOConfig(
    use_vllm=True,
    vllm_mode="colocate",
)
logger.info(f"len(train_dataset): {len(dataset.train_dataset)}")
logger.info(f"len(test_dataset): {len(dataset.test_dataset)}")

trainer = GRPOTrainer(
    model=model,
    reward_funcs=dataset.reward_functions,
    train_dataset=dataset.train_dataset,
    eval_dataset=dataset.test_dataset,
    args=config,
)
trainer.train()
