from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

torch.cuda.empty_cache()
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

base_model = dir_path + "/fiekmodel"
tokenizer = AutoTokenizer.from_pretrained(base_model)

#Ngarkimi i modelit nga lokalisht
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

model.config.use_cache = True

#Interaksioni me modelin e trajnuar

first_question = "What happened in season 4 of Mr. Robot?"

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": first_question}], tokenize=False, add_generation_prompt=True
)
outputs = pipe(prompt, max_new_tokens=400, do_sample=True)

# print(outputs)
print(outputs[0]["generated_text"])

#
second_question = "What is Dissociative identity disorder?"

prompt = tokenizer.apply_chat_template(
    [
        {"role": "system",
         "content": "You show only the direct answer, do not say here is what you have asked for or anything like that."},
        {"role": "user",
         "content": "Someone first asked \"" + first_question + "\", and then followed up with question \"" + second_question + "\". answer the second question and relate it to the first question."
         # "content": "Someone asked \"What happened in season 4 of Mr. Robot?\" and then asked \"What is Dissociative identity disorder?\". Generate a question that includes all the questions asked for, give importance to the latter question."
},
    ], tokenize=False, add_generation_prompt=True
)

outputs = pipe(prompt, max_new_tokens=400, do_sample=True)

# print(outputs)
print(outputs[0]["generated_text"])