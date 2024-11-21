from transformers import LongT5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset

train_dataset = Dataset.from_json("train_data.json")
test_dataset = Dataset.from_json("test_data.json")

model_name = "google/long-t5-tglobal-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = LongT5ForConditionalGeneration.from_pretrained(model_name)


def preprocess_function(examples):
    inputs = tokenizer(examples["content"], max_length=16384, truncation=True, return_tensors="pt")
    targets = tokenizer(examples["target_summary"], max_length=16384, truncation=True, return_tensors="pt")
    return {
        "input_ids": inputs["input_ids"][0],
        "attention_mask": inputs["attention_mask"][0],
        "labels": targets["input_ids"][0],
    }


train_dataset = train_dataset.map(preprocess_function)
test_dataset = test_dataset.map(preprocess_function)


train_dataset = train_dataset.remove_columns(["content", "target_summary", "title"])
test_dataset = test_dataset.remove_columns(["content", "target_summary", "title"])

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

trainer.train()