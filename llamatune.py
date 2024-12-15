from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer
import torch


# base_model = "/kaggle/input/llama-3.2/transformers/3b-instruct/1"
base_model = "/home/avdyl/FIEKMASTER/llama-3.2-transformers-1b-instruct-v1"

tokenizer = AutoTokenizer.from_pretrained(base_model)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    return_dict=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)


if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

messages = [{"role": "user", "content": "what nationality is Rita Ora?"}]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

outputs = pipe(prompt, max_new_tokens=120, do_sample=True)

print(outputs[0]["generated_text"])
