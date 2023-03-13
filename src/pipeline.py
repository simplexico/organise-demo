from pathlib import Path

import typer

from prepare_corpus import prepare_corpus
from train_doc2vec_model import train
from evaluate_model import evaluate
from utils import TRAIN_DATA_PATH, MODELS_PATH, get_latest_from_path


def run_pipeline(txt_files_dir: Path, model_config_path: Path):

    prepare_corpus(txt_files_dir, TRAIN_DATA_PATH)
    just_created_corpus_path = get_latest_from_path(TRAIN_DATA_PATH)

    train(just_created_corpus_path, model_config_path, MODELS_PATH)
    just_created_model_path = get_latest_from_path(MODELS_PATH)

    evaluate(just_created_corpus_path, just_created_model_path)


if __name__ == "__main__":
    typer.run(run_pipeline)
