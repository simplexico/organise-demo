from pathlib import Path
from collections import Counter

import typer
from tqdm import tqdm
from gensim.models import Doc2Vec

from utils import infer_most_similar, load_corpus


def self_similarity_evaluation(train_corpus, model):
    ranks = []
    second_ranks = []
    for doc_id in tqdm(range(len(train_corpus))):
        query = " ".join(train_corpus[doc_id].words)
        sims = infer_most_similar(query, model, topn=len(model.dv))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)

        second_ranks.append(sims[1])
    return ranks, second_ranks


def evaluate(corpus_path: Path, model_path: Path):
    typer.echo("Evaluating model...")
    typer.echo(f"Loading corpus from {corpus_path}...")
    train_corpus = load_corpus(corpus_path)

    typer.echo(f"Loading model from {model_path}...")
    model = Doc2Vec.load(str(model_path))

    typer.echo("Performing self similarity evalution")
    ranks, second_ranks = self_similarity_evaluation(train_corpus, model)

    rank_counts = Counter(ranks)
    n = sum(rank_counts.values())
    for i in range(len(rank_counts)):
        typer.echo(f"{i}'th position ranks")
        v = rank_counts[i]
        percentage = (v / n) * 100
        typer.echo(f"{percentage} %")


if __name__ == "__main__":
    typer.run(evaluate)
