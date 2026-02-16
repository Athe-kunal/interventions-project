from vllm import ModelRegistry
from .qwen2_vllm import Qwen2InterventionsForCausalLM
from .qwen3_vllm import Qwen3InterventionsForCausalLM
from .llama_vllm import LlamaInterventionsForCausalLM

ModelRegistry.register_model(
    "Qwen2InterventionsForCausalLM",
    Qwen2InterventionsForCausalLM,
)
ModelRegistry.register_model(
    "Qwen3InterventionsForCausalLM",
    Qwen3InterventionsForCausalLM,
)
ModelRegistry.register_model(
    "LlamaInterventionsForCausalLM",
    LlamaInterventionsForCausalLM,
)
