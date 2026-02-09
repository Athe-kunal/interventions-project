#!/usr/bin/env python3
"""
Example script for loading and using a trained model with interventions.

Usage:
    python example_inference.py \
        --model_dir "./outputs/experiment/final_model" \
        --prompt "What is 2 + 2?"
"""

import argparse
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig

from interventions_rl.model import qwen3, llama, interventions_utils


def load_model_with_interventions(model_dir: str, device: str = "cuda"):
    """Load a trained model with interventions from a directory"""
    model_path = Path(model_dir)
    
    # Load interventions config
    config_path = model_path / "interventions_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Interventions config not found at: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    interventions_config = interventions_utils.InterventionsConfig(**config_dict)
    print(f"Loaded interventions config: {interventions_config}")
    
    # Load training config to get model name
    training_config_path = model_path.parent / "training_config.yaml"
    if training_config_path.exists():
        import yaml
        with open(training_config_path, 'r') as f:
            training_config = yaml.safe_load(f)
        model_name = training_config['model']['model_name_or_path']
    else:
        # Fallback: ask user or use default
        model_name = input("Enter the original model name (e.g., Qwen/Qwen3-1.7B): ")
    
    print(f"Base model: {model_name}")
    
    # Determine model class
    if "qwen" in model_name.lower():
        model_class = qwen3.Qwen3ForCausalLM
    elif "llama" in model_name.lower():
        model_class = llama.LlamaForCausalLM
    else:
        raise ValueError(f"Could not determine model class for: {model_name}")
    
    # Load HF config
    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    # Create model with interventions
    model = model_class(
        interventions_config=interventions_config,
        config=hf_config,
    )
    
    # Load trained weights
    weights_path = model_path / "model.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")
    
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model weights from: {weights_path}")
    
    # Load tokenizer
    tokenizer_path = model_path / "tokenizer"
    if tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        # Fallback to base model tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    print("Model loaded successfully!")
    
    return model, tokenizer, interventions_config


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda",
):
    """Generate a response using the model"""
    
    # Add system prompt if needed
    from interventions_rl.data.system_prompts import SYSTEM_PROMPT
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    
    # Format with chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1]
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0]
    
    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="Inference with trained interventions model")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the final_model directory (e.g., ./outputs/experiment/final_model)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is 2 + 2?",
        help="Prompt to generate response for",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum generation length",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Loading model with interventions...")
    print("=" * 80)
    
    # Load model
    model, tokenizer, interventions_config = load_model_with_interventions(
        args.model_dir,
        device=args.device,
    )
    
    print("\n" + "=" * 80)
    print("Model ready for inference!")
    print("=" * 80)
    print(f"Interventions: {interventions_config.intervention_type}")
    print(f"Layers: {interventions_config.intervention_layers}")
    print(f"Rank: {interventions_config.low_rank_dimension}")
    print("=" * 80 + "\n")
    
    if args.interactive:
        # Interactive mode
        print("Interactive mode. Type 'quit' or 'exit' to stop.\n")
        
        while True:
            prompt = input("User: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not prompt:
                continue
            
            print("\nGenerating response...\n")
            
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                device=args.device,
            )
            
            print(f"Assistant: {response}\n")
            print("-" * 80 + "\n")
    
    else:
        # Single prompt mode
        print(f"Prompt: {args.prompt}\n")
        print("Generating response...\n")
        
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
        )
        
        print("=" * 80)
        print("RESPONSE:")
        print("=" * 80)
        print(response)
        print("=" * 80)


if __name__ == "__main__":
    main()
