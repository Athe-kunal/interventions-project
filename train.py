import torch
import os
import sys

from typing import Optional
from transformers import set_seed, AutoTokenizer
from trl import GRPOConfig
from loguru import logger

from config import TrainConfig
from interventions_rl.data.open_r1 import load_openr1_dataset
from interventions_rl.model import qwen, llama, interventions_utils
from interventions_rl.model.load_model import load_interventions_model
from interventions_rl.trainer.grpo_trainer import InterventionsGRPOTrainer


_logged: set[str] = set()


def init_logger() -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="[Interventions] {time:YYYY-MM-DD HH:mm:ss} - {name} - {level} - {message}",
    )
    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"


def warn_once(msg: str) -> None:
    """Log a warning message only once per unique message."""
    if msg not in _logged:
        logger.warning(msg)
        _logged.add(msg)


def fuzzy_jobs(args: TrainConfig):
    init_logger()
    args.training.output_dir = args.training.output_dir or "output"
    args.training.run_name = (
        args.training.run_name or args.training.output_dir
    )  # training run name is the output_dir
    if not os.path.exists(args.training.output_dir):  # check if output_dir exists
        os.makedirs(args.training.output_dir, exist_ok=True)
    else:
        logger.info(
            f"Output directory {args.training.output_dir} already exists, using it"
        )
    set_seed(args.common.seed)

    if args.common.debug:
        args.training.report_to = []

    # only initialize for rank 0 when process group is available
    is_main_process = True
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        is_main_process = torch.distributed.get_rank() == 0

    if is_main_process:
        if "trackio" in args.training.report_to:
            import trackio

            trackio.init(
                project=args.logging.trackio_project,
                space_id=args.logging.trackio_space_id,
                config=vars(args.training),
            )
            logger.info(f"Trackio initialized successfully")
        elif "wandb" in args.training.report_to:
            import wandb

            wandb.init(
                name=args.training.run_name,
                config=vars(args.training),
            )
            logger.info(f"Wandb initialized successfully")

    return args


def train(config: Optional[TrainConfig] = None):
    # 0. parse args and prepare logger
    print(config)
    args = fuzzy_jobs(config)

    # 1. load tokenizer and dataset
    logger.info(f"Loading tokenizer from {args.model.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model.model_name_or_path)
    tokenizer.padding_side = (
        "left"  # Configure for decoder-only architecture: use left padding
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token if tokenizer.eos_token is not None else "<|endoftext|>"
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
            if tokenizer.eos_token_id is not None
            else tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        )

    logger.info(f"Loading dataset from {args.dataset.dataset_name_or_path}")
    dataset = load_openr1_dataset(
        args.dataset.dataset_name_or_path,
        example_numbers=args.dataset.example_numbers,
    )
    train_dataset = dataset.train_dataset
    reward_functions = dataset.reward_functions

    reward_weights = dataset.reward_weights or [1.0] * len(reward_functions)
    args.training.reward_weights = reward_weights

    # 2. load model with interventions
    logger.info(f"Loading model from {args.model.model_name_or_path}")
    dtype = torch.bfloat16 if args.model.dtype == "bfloat16" else torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build interventions config from args
    ic = args.interventions
    iv_config = interventions_utils.InterventionsConfig(
        intervention_type=ic.intervention_type,
        intervention_layers=ic.intervention_layers,
        low_rank_dimension=ic.low_rank_dimension,
        dropout=ic.dropout,
        act_fn=ic.act_fn,
        init_orth=ic.init_orth,
    )

    # Infer model class from name
    model_name = args.model.model_name_or_path.lower()
    if "qwen3" in model_name:
        model_class = qwen.Qwen3ForCausalLM
    elif "qwen2" in model_name:
        model_class = qwen.Qwen2ForCausalLM
    elif "llama" in model_name:
        model_class = llama.LlamaForCausalLM
    else:
        raise ValueError(
            f"Cannot infer model type from: {args.model.model_name_or_path}"
        )

    report, model = load_interventions_model(
        hf_model_name_or_path=args.model.model_name_or_path,
        model_class=model_class,
        ic_config=iv_config,
        map_dtype=dtype,
        map_device=device,
    )
    logger.info(f"Model loaded: {report.summary()}")

    # 3.Training configuration
    training_args = GRPOConfig(
        **vars(args.training),
    )

    # 4.Train
    logger.info(f"Training model with GRPO")
    trainer = InterventionsGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_dataset,
    )

    resume_checkpoint = args.training.resume_from_checkpoint
    if resume_checkpoint == "true":
        resume_checkpoint = True
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    logger.info(f"Training completed successfully")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
