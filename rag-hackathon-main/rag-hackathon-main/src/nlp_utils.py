"""Shared spaCy pipeline loader.

Loaded once per process and reused everywhere (sentence splitting in
chunking.py, lemmatization-based keyword matching in retrieval.py)
instead of each module loading its own copy of the model.
"""

from __future__ import annotations

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            _nlp = spacy.blank("en")
        if "sentencizer" not in _nlp.pipe_names and "parser" not in _nlp.pipe_names:
            _nlp.add_pipe("sentencizer")
    return _nlp