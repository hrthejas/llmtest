import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    LlamaTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer
)
from getpass import getpass
from datasets import load_dataset


def login_to_hub():
    from huggingface_hub import login
    token = getpass("Enter hf token")
    login(token=token)


def load_model(model_name):
    conf = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=conf,
        device_map="cuda",
        use_auth_token=True,
        trust_remote_code=True,
        # from_tf=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def generate_prompt(data_point):
    return f"""
  Below is an instruction that describes a task. Write a response that appropriately completes the request.
  ### INSTRUCTION: 
  {data_point["instruction"]}

  ### RESPONSE: 
  {data_point["output"]}
  """.strip()


def generate_and_tokenize_prompt(data_point, tokenizer):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenizer(full_prompt, padding=True)  # truncation=True,max_length=2048
    return tokenized_full_prompt


def dataset_for_training(tokenizer):
    tokenizer.pad_token = tokenizer.eos_token

    train_data = load_dataset("ehartford/WizardLM_alpaca_evol_instruct_70k_unfiltered")
    load_dataset("json",data_files=)
    train_datatest = train_data["train"].shuffle().map(generate_and_tokenize_prompt)

    import random
    n = 5000
    eval_datatest = train_datatest.select(random.sample(range(len(train_datatest)), n))

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    return train_datatest, eval_datatest, data_collator


def get_trainser_args():
    training_args = TrainingArguments(
        output_dir="/notebooks/models/bllom-tiny-v4",  # The output directory
        overwrite_output_dir=True,  # overwrite the content of the output directory
        num_train_epochs=4,  # number of training epochs
        per_device_train_batch_size=6,  # batch size for training
        per_device_eval_batch_size=2,  # batch size for evaluation
        eval_steps=1000,  # Number of update steps between two evaluations.
        save_steps=1000,  # after # steps model is saved
        warmup_steps=30,  # number of warmup steps for learning rate scheduler
        prediction_loss_only=True,
        logging_steps=1,
        learning_rate=2e-4,
        max_steps=-1,
        dataloader_drop_last=True,
        logging_dir="/notebooks/models/logs",
        gradient_checkpointing=True,
        torch_compile=False,
        gradient_accumulation_steps=10,
        optim="adamw_torch",
        save_strategy="steps",
        evaluation_strategy='steps',
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False,
        save_total_limit=3,
        fp16=False,  # If model is loaded using torch_dtype=torch.float16 tthis should be set to False
        bf16=False,  # If model is loaded using torch_dtype=torch.float16 this should be set to true
        report_to=["tensorboard"],
        no_cuda=False,
    )
    return training_args

def train(model,tokenizer,training_args,train_dataset,eval_dataset,data_collector):
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collector,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
