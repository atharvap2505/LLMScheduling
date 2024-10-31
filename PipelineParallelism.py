import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import distributed as dist
import deepspeed

# Initialize the distributed process group
dist.init_process_group(backend='nccl', init_method='env://')

# Check rank and world size for logging and partitioning
rank = int(os.getenv('RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

# Define the model name and tokenizer
model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with deepspeed for pipeline parallelism across containers
model = AutoModelForCausalLM.from_pretrained(model_name)

# DeepSpeed configuration for ZeRO optimization and pipeline parallelism
ds_config = {
    "train_batch_size": 1,
    "train_micro_batch_size_per_gpu": 1,
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,  # ZeRO stage 3 for full model parallelism
        "offload_param": {
            "device": "cpu"  # Offload parameters to CPU to save GPU memory
        },
    },
    "pipeline": {
        "enable": True,
        "stages": world_size  # Number of stages equal to number of containers
    }
}

# Initialize model with DeepSpeed and capture all returned values explicitly
model, optimizer, dataloader, engine = deepspeed.initialize(
    model=model,
    config_params=ds_config,
    model_parameters=None
)

# Define the input text and tokenize it
input_text = "What is the capital of France?"
inputs = tokenizer(input_text, return_tensors="pt").input_ids.to(rank)  # Place inputs on correct GPU

# Perform inference with no_grad
with torch.no_grad():
    outputs = model.generate(inputs)

# Decode and print the output on rank 0
if rank == 0:
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Output from rank {rank}: {output_text}")

# Clean up the process group
dist.destroy_process_group()
