import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer, TrainingArguments
from datasets import load_dataset
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
torch.cuda.empty_cache()

gc.collect()
from trl import SFTTrainer, setup_chat_format
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import wandb
from huggingface_hub import login

hf_token = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
login(token = hf_token)

wb_token = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

wandb.login(key=wb_token)
run = wandb.init(
    project='Fine-tune Llama 3 8B on Medical Dataset',
    job_type="training",
    anonymous="allow"
)


base_model = "/home/avdyl/FIEKMASTER/llama-3.2-transformers-1b-instruct-v1"
tokenizer = AutoTokenizer.from_pretrained(base_model)



#Import dataset
dataset = load_dataset('json', data_files='./contextual_pairing.json', split='all')

def format_chat_template(row):
    print(row)
    row_json = [{"role": "user", "content": row['0']},
               {"role": "assistant", "content": row['1']}]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row



dataset = dataset.map(format_chat_template)
#
#
#
# # base_model = "/kaggle/input/llama-3.2/transformers/3b-instruct/1"
new_model = "./fiekmodel"

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    return_dict=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)


peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)
model = get_peft_model(model, peft_config)


if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id




dataset = dataset.train_test_split(test_size=0.2)

training_arguments = TrainingArguments(
    output_dir=new_model,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    evaluation_strategy="steps",
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=1,
    logging_strategy="steps",
    learning_rate=2e-4,
    # learning_rate=100000,
    fp16=False,
    bf16=False,
    group_by_length=True,
    report_to="none",
    # report_to="wandb"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    # max_seq_length=512,
    # dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    # packing= False,
)

wandb.finish()

trainer.train()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)


model.config.use_cache = True
