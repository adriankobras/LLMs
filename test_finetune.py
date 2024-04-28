import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PEFT

# Load pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct", 
    device_map="mps", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

# Load and tokenize training data
train_file = "lyrics.txt"
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=128  # adjust as needed
)

# Define the PEFT model
model = PEFT(model)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,  # reduce batch size
    save_steps=10_000,
    save_total_limit=2,
    #fp16=True,  # use mixed precision
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    ),
    train_dataset=train_dataset,
    lora=lora,  # Use LORA for fine-tuning
)

# Fine-tune the model
trainer.train()

# Example text generation
prompt = tokenizer.bos_token  # Start of sequence token

# Tokenize prompt and move input to the MPS device
input_ids = tokenizer.encode(prompt, return_tensors="pt").to('mps')

# Generate text
generated = model.generate(
    input_ids,
    max_length=100,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id,
)

# Decode and print generated text
print(tokenizer.decode(generated[0], skip_special_tokens=True))
