import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import distributed as dist
import deepspeed

# Initialize the distributed process group with GPU-optimised backend
dist.init_process_group(backend='nccl', init_method='env://')

# Check rank and world size for logging and partitioning, will be fetched from env file
# I've defaulted it to 0 and 1, but can be changed depending on how many containers we have
rank = int(os.getenv('RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# DeepSpeed configuration for ZeRO optimization and pipeline parallelism
ds_config = {
    "train_batch_size": 1,
    "train_micro_batch_size_per_gpu": 1,
    "fp16": {
        "enabled": True # Can be disabled if precision is expected.
    },
    "zero_optimization": {
        "stage": 3,  # ZeRO stage 3 for full model parallelism
        "offload_param": {
            "device": "cpu"  # Offload parameters to CPU to save GPU memory
        },
    },
    "pipeline": { # Implementation of pipeline parallelism
        "enable": True,
        "stages": world_size  
    }
}

# Initialize model with DeepSpeed and capture all returned values explicitly
model, optimizer, dataloader, engine = deepspeed.initialize(
    model=model,
    config_params=ds_config,
    model_parameters=None
)

input_text = "What is the capital of France?"
inputs = tokenizer(input_text, return_tensors="pt").input_ids.to(rank)  # Place inputs on correct GPU

# Prevent redundant gradient calculations for optimised execution
with torch.no_grad():
    outputs = model.generate(inputs)

# Decode and print the output on rank 0 to avoid multiple/repeated outputs
if rank == 0:
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Output from rank {rank}: {output_text}")

# Clean up the process group
dist.destroy_process_group()
