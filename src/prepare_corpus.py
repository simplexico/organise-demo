import typer
from pathlib import Path
from typing import List, Dict

import cloudpickle
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess

from utils import get_datetime_str


def process_tokenize_text(text: str) -> List[str]:
    """Process text and tokenize"""
    tokens = simple_preprocess(text)
    return tokens


def load_txt_files(txt_files_dir: Path) -> Dict:
    corpus = {}
    for txt_file in sorted(txt_files_dir.iterdir()):
        with txt_file.open() as f:
            txt = f.read().split()
        filename = txt_file.stem.strip()
        corpus[filename] = ' '.join(txt)
    return corpus


def prepare_corpus(txt_files_dir: Path, output_dir: Path):
    typer.echo('Preparing corpus...')

    typer.echo(f'Loading text files from {txt_files_dir}...')
    corpus = load_txt_files(txt_files_dir)

    train_corpus = []
    for doc_id, filename in enumerate(corpus.keys()):
        doc = corpus[filename]
        tokens = process_tokenize_text(doc)
        tagged_doc = TaggedDocument(tokens, tags=[doc_id, filename])
        train_corpus.append(tagged_doc)

    typer.echo(f"Writing corpus to {output_dir}...")
    output_path = output_dir / f"train-corpus-{get_datetime_str()}.pkl"
    output_path.touch(exist_ok=True)

    with output_path.open('wb') as f:
        cloudpickle.dump(train_corpus, f)


if __name__ == "__main__":
    typer.run(prepare_corpus)
