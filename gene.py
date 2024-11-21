import modal
from transformers import LongT5ForConditionalGeneration, T5Tokenizer
from datasets import Dataset
import json

app = modal.App(name="long-t5-summary-generation")

@app.function(
    image=modal.Image.debian_slim()
    .pip_install("torch", "transformers[torch]", "datasets", "sentencepiece", "numpy"),
    gpu="A10G",
    timeout=3600,
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")],
    volumes={"/root/results": modal.Volume.from_name("my-volume")}
)
def generate_summary():
    test_dataset = Dataset.from_json("/root/test_data.json")

    model_dir = "/root/results/checkpoint-120"
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = LongT5ForConditionalGeneration.from_pretrained(model_dir)

    summaries = []

    for example in test_dataset:
        content = example["content"]

        def chunk_text(text, num_chunks=3):
            tokens = tokenizer.encode(text)
            chunk_size = len(tokens) // num_chunks
            return [tokenizer.decode(tokens[i * chunk_size: (i + 1) * chunk_size]) for i in range(num_chunks)]

        content_chunks = chunk_text(content, num_chunks=3)
        chunk_summaries = []
        for chunk in content_chunks:
            inputs = tokenizer(chunk, return_tensors="pt", max_length=8192, truncation=True)
            summary_ids = model.generate(inputs["input_ids"], max_length=256, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            chunk_summaries.append(summary)

        full_summary = " ".join(chunk_summaries)

        summaries.append({
            "title": example.get("title", "No Title"),
            "content": content,
            "generated_summary": full_summary
        })

    output_file = "/root/results/generated_summaries.json"
    with open(output_file, "w") as f:
        json.dump(summaries, f, indent=4)

    print(f"Generated summaries saved to {output_file}")


if __name__ == "__main__":
    with app.run():
        generate_summary()