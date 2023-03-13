import os
from pathlib import Path
from typing import List, Dict

import typer
import yaml
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from utils import get_datetime_str, load_corpus


def load_yaml_file(path: Path) -> Dict[str, int]:
    typer.echo('Loading config...')
    with open(path) as f:
        data = yaml.safe_load(f)
    return data


def train_doc2vec_model(train_corpus: List[TaggedDocument], model_params: Dict[str, int]) -> Doc2Vec:
    model = Doc2Vec(**model_params, workers=os.cpu_count())
    typer.echo('Building vocab....')
    model.build_vocab(train_corpus)
    typer.echo('Training model....')
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    return model


def train(corpus_path: Path, model_config_path: Path, output_dir: Path):
    typer.echo("Training model...")
    typer.echo(f"Loading corpus from {corpus_path}...")
    train_corpus = load_corpus(corpus_path)

    model_params = load_yaml_file(model_config_path)
    model = train_doc2vec_model(train_corpus, model_params)
    typer.echo(f'Saving model to {output_dir}...')
    output_path = output_dir / f"doc2vec-{get_datetime_str()}.model"
    model.save(str(output_path))


if __name__ == "__main__":
    typer.run(train)
