import json
import google.generativeai as genai

genai.configure(api_key='AIzaSyBhQh_KpObWgXhgoyjhi0ZtjySOrZlHFr8')
model = genai.GenerativeModel('gemini-pro')

input_file_path = 'random_bio_articles.json'

def read_content_from_txt(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading text file {txt_path}: {e}")
        return ""

def print_gemini_response(title, content_text):
    prompt = (
        f"Please carefully read the following content of a biology research article titled '{title}'. "
        f"After reading, provide two ratings:\n\n"
        f"1. Comprehensibility: Rate how easy it is to understand the content on a scale from 1 to 5 "
        f"(1 = very hard to understand, 5 = very easy to understand).\n"
        f"2. Flesch-Kincaid Readability Score: Provide the readability score for the content.\n\n"
        f"Content:\n{content_text}\n\n"
    )

    messages = [{'role': 'user', 'parts': prompt}]

    response = model.generate_content(
        messages,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            temperature=0.7
        )
    )

    print("Response received:", response)

with open(input_file_path, 'r') as file:
    articles = json.load(file)

for article in articles[:2]:
    txt_path = f"dataset_code/{article['content']}"
    content_text = read_content_from_txt(txt_path)

    if content_text.strip():
        print_gemini_response(article["title"], content_text)