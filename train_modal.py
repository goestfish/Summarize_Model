import modal
from transformers import LongT5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import json

app = modal.App(name="long-t5-training")

@app.function(
    image=modal.Image.debian_slim()
    .pip_install("torch", "transformers[torch]", "datasets", "sentencepiece", "numpy"),
    gpu="A10G",
    timeout=3600,
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")],
    volumes={"/root/results": modal.Volume.from_name("my-volume")}
)
def train_and_gene():
    train_dataset = Dataset.from_json("/root/train_data.json")
    test_dataset = Dataset.from_json("/root/test_data.json")

    model_name = "google/long-t5-tglobal-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = LongT5ForConditionalGeneration.from_pretrained(model_name)

    def chunk_text(text, num_chunks=3):
        tokens = tokenizer.encode(text)
        chunk_size = len(tokens) // num_chunks
        return [tokenizer.decode(tokens[i * chunk_size: (i + 1) * chunk_size]) for i in range(num_chunks)]

    def preprocess_function(examples):
        content_chunks = chunk_text(examples["content"])
        inputs = [
            tokenizer(chunk, max_length=8192, truncation=True, return_tensors="pt") for chunk in content_chunks
        ]
        targets = tokenizer(examples["target_summary"], max_length=8192, truncation=True, return_tensors="pt")
        return {
            "input_ids": [chunk["input_ids"][0] for chunk in inputs],
            "attention_mask": [chunk["attention_mask"][0] for chunk in inputs],
            "labels": targets["input_ids"][0],
        }

    train_dataset = train_dataset.map(preprocess_function)
    test_dataset = test_dataset.map(preprocess_function)

    train_dataset = train_dataset.map(lambda example: {
        "input_ids": example["input_ids"][0],
        "attention_mask": example["attention_mask"][0],
        "labels": example["labels"]
    })
    test_dataset = test_dataset.map(lambda example: {
        "input_ids": example["input_ids"][0],
        "attention_mask": example["attention_mask"][0],
        "labels": example["labels"]
    })

    training_args = TrainingArguments(
        output_dir="/root/results",
        eval_strategy="epoch",
        learning_rate=0.00005,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="/root/logs",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    summaries = []

    for example in test_dataset:
        title = example.get("title", "No Title")
        content_chunks = chunk_text(example["content"], num_chunks=3)
        chunk_summaries = []

        for chunk in content_chunks:
            model = model.to("cuda")
            inputs = tokenizer(chunk, return_tensors="pt", max_length=8192, truncation=True)
            inputs = {key: value.to("cuda") for key, value in inputs.items()}

            summary_ids = model.generate(inputs["input_ids"], max_length=256, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            chunk_summaries.append(summary)

        full_summary = " ".join(chunk_summaries)
        summaries.append({
            "title": title,
            "generated_summary": full_summary
        })

    # 保存摘要到 JSON 文件
    output_file = "/root/results/generated_summaries.json"
    with open(output_file, "w") as f:
        json.dump(summaries, f, indent=4)

    print(f"Generated summaries saved to {output_file}")


if __name__ == "__main__":
    with app.run():
        train_and_gene()