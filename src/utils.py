from pathlib import Path
from datetime import datetime
from typing import List

import cloudpickle
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess

MODELS_PATH = Path("models/")
RAW_DATA_PATH = Path("data/0_raw_data/")
TRAIN_DATA_PATH = Path("data/1_train_data/")


def get_latest_from_path(path: Path, pattern: str = "*"):
    files = path.glob(pattern)
    return max(files, key=lambda x: x.stat().st_ctime)


def get_datetime_str() -> str:
    return datetime.now().strftime('%d-%m-%Y-%H-%M')


def load_corpus(corpus_path: Path) -> List[TaggedDocument]:
    with corpus_path.open('rb') as f:
        train_corpus = cloudpickle.load(f)
    return train_corpus


def infer_most_similar(query: str, model: Doc2Vec, topn: int = 3):
    tokens = simple_preprocess(query)
    inferred_vector = model.infer_vector(tokens)
    sims = model.dv.most_similar([inferred_vector], topn=topn)
    return sims
