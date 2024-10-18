# Load model 
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import no_grad
from accelerate import init_empty_weights, infer_auto_device_map

model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize empty model and infer device map
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(model_name)

device_map = infer_auto_device_map(model, max_memory={0: "14GiB", 1: "14GiB"})

# Load the model manually onto the devices in the map
for device, layers in device_map.items():
    for layer_name in layers:
        getattr(model, layer_name).to(f'cuda:{device}')

# Tokenize input
input_text = "What is the capital of France?"
inputs = tokenizer(input_text, return_tensors="pt").input_ids

# Move inputs to the first device in the device map
device = list(device_map.keys())[0]
inputs = inputs.to(f'cuda:{device}')

# Run the model and generate output
with no_grad():
    outputs = model.generate(inputs)

# Decode output
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)
