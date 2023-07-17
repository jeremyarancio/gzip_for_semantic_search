
import numpy as np
from tqdm import tqdm

from datasets import load_dataset

from semantic_search_algorithms import GzipSemanticSearch, SBert


if __name__ == "__main__":

    dataset = load_dataset(path="Hello-SimpleAI/HC3", name="all", split="train[:100]")
    dataset = dataset.shuffle(seed=42)
    questions = dataset["question"]
    human_answers = dataset["human_answers"]
    sbert = SBert()
    
    gzip_results = []
    sbert_results = []

    for i_quest, question in tqdm(enumerate(questions), total=len(questions)):
        gzip_distances = []
        sbert_distances = []

        for answer in tqdm(human_answers):
            gzip_distances.append(GzipSemanticSearch.compute_distance(question, answer[0]))
            sbert_distances.append(sbert.compute_distance(question, answer[0]))

        gzip_result = np.argmin(gzip_distances)
        sbert_result = np.argmax(sbert_distances) #cos similarity

        gzip_results.append(gzip_result == i_quest)
        sbert_results.append(sbert_result == i_quest)

    gzip_accuracy = np.sum(gzip_results) / len(gzip_results)
    sbert_accuracy = np.sum(sbert_results) / len(sbert_results)

    print(f"Gzip accuracy: {gzip_accuracy}", file=open("results.txt", "a"))
    print(f"SBert accuracy: {sbert_accuracy}", file=open("results.txt", "a"))
    
