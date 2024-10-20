from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import load_checkpoint_and_dispatch
from torch import no_grad

# Define the model name and tokenizer
model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Automatically distribute the model across multiple GPUs using Accelerate
model = load_checkpoint_and_dispatch(model, device_map="auto")

# Prepare and tokenize input text
input_text = "What is the capital of France?"
inputs = tokenizer(input_text, return_tensors="pt").input_ids

# Perform inference with no_grad to avoid unnecessary gradient calculations
with no_grad():
    outputs = model.generate(inputs)

# Decode the output tokens into a human-readable string
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the output
print(output_text)
