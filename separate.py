import json
from sklearn.model_selection import train_test_split


with open("dataset_code/rated_bio_articles.json", "r") as file:
    data = json.load(file)


def preprocess_data(data):
    processed_data = []
    for item in data:
        content_path = "dataset_code/" + item["content"]
        with open(content_path, "r") as f:
            content = f.read()
        processed_data.append({
            "content": content,
            "target_summary": f"Please summarize this essay on Biology for high school students, you should assume that high school students "
                              f"do not have the corresponding college knowledge, vocabulary should be brief and clear, and you should keep "
                              f"your summary to about 500 words.",
            "title": item["title"],
        })
    return processed_data

processed_data = preprocess_data(data)

train_data, test_data = train_test_split(processed_data, test_size=0.2, random_state=42)

with open("train_data.json", "w") as file:
    json.dump(train_data, file)
with open("test_data.json", "w") as file:
    json.dump(test_data, file)