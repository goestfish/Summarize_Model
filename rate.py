import time
import json
import google.generativeai as genai

genai.configure(api_key='Api Key')
model = genai.GenerativeModel('gemini-pro')

input_file_path = 'dataset_code/random_bio_articles.json'
output_file_path = 'dataset_code/rated_bio_articles.json'

REQUEST_LIMIT_PER_MINUTE = 3
SECONDS_PER_REQUEST = 60 / REQUEST_LIMIT_PER_MINUTE

def read_content_from_txt(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading text file {txt_path}: {e}")
        return ""

def rate_article_with_gemini(title, content_text):
    prompt = (
        f"Please carefully read the following content of a biology research article titled '{title}'. "
        f"After reading, provide two ratings:\n\n"
        f"1. Comprehensibility: Rate how easy it is to understand the content on a scale from 1 to 5 "
        f"(1 = very hard to understand, 5 = very easy to understand).\n"
        f"2. Flesch-Kincaid Readability Score: Provide the readability score for the content.\n\n"
        f"Content:\n{content_text}\n\n"
        f"Provide the response a a JSON object with two fields: 'rate1' and 'rate2'."
    )

    messages = [{'role': 'user', 'parts': prompt}]

    response = model.generate_content(
        messages,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            temperature=0.7
        )
    )

    if response.candidates and len(response.candidates) > 0:
        candidate = response.candidates[0]

        if hasattr(candidate.content, 'parts') and len(candidate.content.parts) > 0:
            generated_text = candidate.content.parts[0].text
            print("Generated content:", generated_text)

            try:
                cleaned_text = generated_text.strip().replace("```json", "").replace("```", "").strip()
                response_json = json.loads(cleaned_text)
                rate1 = response_json.get("rate1")
                rate2 = response_json.get("rate2")
                return rate1, rate2
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")
                return None, None
        else:
            print("No content found in the candidate.")
    else:
        print("No candidates found in the response.")

    return None, None

with open(input_file_path, 'r') as file:
    articles = json.load(file)

try:
    with open(output_file_path, 'r') as output_file:
        rated_articles = json.load(output_file)
        processed_titles = {article["title"] for article in rated_articles}  # Track already processed articles
except FileNotFoundError:
    rated_articles = []
    processed_titles = set()

try:
    for i, article in enumerate(articles):
        if article["title"] in processed_titles:
            print(f"Skipping already processed article: {article['title']}")
            continue

        txt_path = f"dataset_code/{article['content']}"
        content_text = read_content_from_txt(txt_path)

        if content_text.strip():
            rate1, rate2 = rate_article_with_gemini(article["title"], content_text)

            if rate1 is not None and rate2 is not None:
                article["rate1"] = rate1
                article["rate2"] = rate2

        rated_articles.append(article)
        processed_titles.add(article["title"])

        with open(output_file_path, 'w') as output_file:
            json.dump(rated_articles, output_file, indent=4)

        if (i + 1) % REQUEST_LIMIT_PER_MINUTE == 0:
            print(f"Reached {REQUEST_LIMIT_PER_MINUTE} requests. Waiting for 1 minute...")
            time.sleep(60)
        else:
            time.sleep(SECONDS_PER_REQUEST)

except Exception as e:
    print(f"An error occurred: {e}")

with open(output_file_path, 'w') as output_file:
    json.dump(rated_articles, output_file, indent=4)

print(f"Ratings have been added and saved to {output_file_path}.")
