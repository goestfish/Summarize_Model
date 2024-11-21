import sys
sys.path.append("/root/project")

from modal_env import app
@app.function()
def main():
    from data_preparation import prepare_data
    import os
    print(os.listdir("/root/project"))  # Debugging
    prepare_data("dataset_code/rated_bio_articles.json", "formatted_dataset.json")

