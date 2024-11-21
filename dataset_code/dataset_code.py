import requests
import xmltodict
import json
import random

def fetch_biology_articles(num_articles=50):
    # Base URL for arXiv API
    base_url = "http://export.arxiv.org/api/query"
    articles = []
    unique_ids = set()

    results_per_request = 10

    # Subject area specifically for biology-related topics
    biology_subjects = ["q-bio", "bio", "genomics", "biochemistry", "neuroscience"]

    while len(articles) < num_articles:
        # Choose a random subject area within biology
        subject = random.choice(biology_subjects)
        start_index = random.randint(0, 1000)  # Randomize starting index for variety
        params = {
            "search_query": f"cat:{subject}",
            "start": start_index,
            "max_results": results_per_request
        }

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            data = xmltodict.parse(response.text)
            if "feed" in data and "entry" in data["feed"]:
                entries = data["feed"]["entry"]
                if not isinstance(entries, list):
                    entries = [entries]

                # Add unique articles to the dataset
                for entry in entries:
                    article_id = entry["id"]
                    if article_id not in unique_ids:
                        unique_ids.add(article_id)
                        articles.append(entry)
                        if len(articles) >= num_articles:
                            break
        else:
            print(f"Error: {response.status_code}")

    # Process articles into a structured dataset
    dataset = []
    for article in articles:
        try:
            title = article["title"].strip()
            authors = [a["name"] for a in article["author"]] if isinstance(article["author"], list) else [article["author"]["name"]]
            link = article["id"]
            dataset.append({
                "title": title,
                "authors": authors,
                "link": link,
                "content": ""
            })
        except Exception as e:
            print(f"Error processing article: {e}")
    
    return dataset

# Fetch 50 biology-related articles
biology_articles = fetch_biology_articles(num_articles=50)

with open("random_bio_articles.json", "w") as f:
    json.dump(biology_articles, f, indent=4)

print("Saved 50 random biology-related articles to random_bio_articles.json.")

