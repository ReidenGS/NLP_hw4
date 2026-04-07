import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation
    text = example["text"]

    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = text.split()

    keyboard_neighbors = {
        "a": ["q", "w", "s", "z"], "b": ["v", "g", "h", "n"], "c": ["x", "d", "f", "v"],
        "d": ["s", "e", "r", "f", "c", "x"], "e": ["w", "s", "d", "r"], "f": ["d", "r", "t", "g", "v", "c"],
        "g": ["f", "t", "y", "h", "b", "v"], "h": ["g", "y", "u", "j", "n", "b"], "i": ["u", "j", "k", "o"],
        "j": ["h", "u", "i", "k", "n", "m"], "k": ["j", "i", "o", "l", "m"], "l": ["k", "o", "p"],
        "m": ["n", "j", "k"], "n": ["b", "h", "j", "m"], "o": ["i", "k", "l", "p"],
        "p": ["o", "l"], "q": ["w", "a"], "r": ["e", "d", "f", "t"], "s": ["a", "w", "e", "d", "x", "z"],
        "t": ["r", "f", "g", "y"], "u": ["y", "h", "j", "i"], "v": ["c", "f", "g", "b"],
        "w": ["q", "a", "s", "e"], "x": ["z", "s", "d", "c"], "y": ["t", "g", "h", "u"], "z": ["a", "s", "x"]
    }

    protected_words = {
        "not", "no", "never", "nor",
        "is", "are", "was", "were",
        "am", "be", "been", "being"
    }

    def typo_replace(token):
        # 只改纯字母、长度较长的词
        if not token.isalpha() or len(token) <= 3:
            return token

        # 跳过关键否定词和高频功能词，尽量保持标签不变
        if token.lower() in protected_words:
            return token

        # 控制变换强度：不是每个词都改
        if random.random() > 0.28:
            return token

        chars = list(token)

        # 只改中间字符，不改首尾，更像真实 typo
        idxs = [i for i in range(1, len(chars) - 1) if chars[i].lower() in keyboard_neighbors]
        if not idxs:
            return token

        idx = random.choice(idxs)
        original = chars[idx]
        repl = random.choice(keyboard_neighbors[original.lower()])
        chars[idx] = repl.upper() if original.isupper() else repl

        return "".join(chars)

    transformed = [typo_replace(token) for token in tokens]
    example["text"] = TreebankWordDetokenizer().detokenize(transformed)

    ##### YOUR CODE ENDS HERE ######

    return example
