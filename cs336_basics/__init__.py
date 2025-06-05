try:
    import importlib.metadata
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0-dev"


from .bpe_tokenizer import BPETokenizer
from .bpe_word import Word
