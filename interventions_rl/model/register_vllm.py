from vllm import ModelRegistry
from .qwen2_vllm import Qwen2ForCausalLM
from .qwen3_vllm import Qwen3ForCausalLM
from .llama_vllm import LlamaForCausalLM

ModelRegistry.register_model(
    "Qwen2ForCausalLM",
    Qwen2ForCausalLM,
)
ModelRegistry.register_model(
    "Qwen3ForCausalLM",
    Qwen3ForCausalLM,
)
ModelRegistry.register_model(
    "LlamaForCausalLM",
    LlamaForCausalLM,
)
