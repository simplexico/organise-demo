import pytest

from src.prepare_corpus import process_tokenize_text


@pytest.mark.parametrize("text, output",
                         [('This is text', ['this', 'is', 'text'])])
def test_process_tokenize_text(text, output):
    tokens = process_tokenize_text(text)
    assert tokens == output
