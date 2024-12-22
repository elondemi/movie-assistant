import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer, TrainingArguments
from datasets import load_dataset
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
from dotenv import load_dotenv
import wandb
from huggingface_hub import login

#Pastrimi permes garbage collection
torch.cuda.empty_cache()
gc.collect()

#Ngarkimi i env variablave
load_dotenv()

#Importimi i reinforcement learning
from trl import SFTTrainer, setup_chat_format

#importimi i PEFT (Parameter-Efficient Fine-Tuning)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)


hf_token = os.getenv('hf_token')
login(token = hf_token)

wb_token = os.getenv('wb_token')


#Startimi i wandb
wandb.login(key=wb_token)
run = wandb.init(
    project='Fine-tune Llama 3 8B on Medical Dataset',
    job_type="training",
    anonymous="allow"
)

#Ngarkimi i modelit baze
base_model = "/home/avdyl/FIEKMASTER/llama-3.2-transformers-1b-instruct-v1"
tokenizer = AutoTokenizer.from_pretrained(base_model)



#Import dataset
dataset = load_dataset('json', data_files='./contextual_pairing.json', split='all')

#Definimi i template te chat-it
def format_chat_template(row):
    print(row)
    row_json = [{"role": "user", "content": row['0']},
               {"role": "assistant", "content": row['1']}]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row


#Mapimi i secilit rekord ne dataset permes chat templates
dataset = dataset.map(format_chat_template)

new_model = "./fiekmodel"

#Krijimi i modelit te ri
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

#Ndarja e dataset-it 80 per trajnim / 20 per testim
dataset = dataset.train_test_split(test_size=0.2)

# Below is a list of hyperparameters that can be used to optimize the training process:
#
#     output_dir: The output directory is where the model predictions and checkpoints will be stored.
#     num_train_epochs: One training epoch.
#     fp16/bf16: Disable fp16/bf16 training.
#     per_device_train_batch_size: Batch size per GPU for training.
#     per_device_eval_batch_size: Batch size per GPU for evaluation.
#     gradient_accumulation_steps: This refers to the number of steps required to accumulate the gradients during the update process.
#     gradient_checkpointing: Enabling gradient checkpointing.
#     max_grad_norm: Gradient clipping.
#     learning_rate: Initial learning rate.
#     weight_decay: Weight decay is applied to all layers except bias/LayerNorm weights.
#     Optim: Model optimizer (AdamW optimizer).
#     lr_scheduler_type: Learning rate schedule.
#     max_steps: Number of training steps.
#     warmup_ratio: Ratio of steps for a linear warmup.
#     group_by_length: This can significantly improve performance and accelerate the training process.
#     save_steps: Save checkpoint every 25 update steps.
#     logging_steps: Log every 25 update steps.


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
    # report_to="none",
    report_to="wandb"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    # dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    # packing= False,
)

#Startoje trajnimin
trainer.train()

#Tregoj wandb qe te ndalet ne tracking te trajnimit
wandb.finish()

#Ruaj modelin dhe tokenizer lokalisht
tokenizer.save_pretrained(new_model)
model.save_pretrained(new_model)




