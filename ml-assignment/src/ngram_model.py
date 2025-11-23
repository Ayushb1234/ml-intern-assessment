# import random

# class TrigramModel:
#     def __init__(self):
#         """
#         Initializes the TrigramModel.
#         """
#         # TODO: Initialize any data structures you need to store the n-gram counts.
       
#         pass

#     def fit(self, text):
#         """
#         Trains the trigram model on the given text.

#         Args:
#             text (str): The text to train the model on.
#         """
#         # TODO: Implement the training logic.
#         # This will involve:
#         # 1. Cleaning the text (e.g., converting to lowercase, removing punctuation).
#         # 2. Tokenizing the text into words.
#         # 3. Padding the text with start and end tokens.
#         # 4. Counting the trigrams.
#         pass

#     def generate(self, max_length=50):
#         """
#         Generates new text using the trained trigram model.

#         Args:
#             max_length (int): The maximum length of the generated text.

#         Returns:
#             str: The generated text.
#         """
#         # TODO: Implement the generation logic.
#         # This will involve:
#         # 1. Starting with the start tokens.
#         # 2. Probabilistically choosing the next word based on the current context.
#         # 3. Repeating until the end token is generated or the maximum length is reached.
#         pass



import re
import random
from collections import defaultdict, Counter


class TrigramModel:
    def __init__(self, unk_threshold=1):
        self.trigrams = defaultdict(Counter)
        self.bigrams = defaultdict(Counter)
        self.unigrams = Counter()
        self.vocab = set()
        self.unk_threshold = unk_threshold
        self.total_unigrams = 0

    def _split_sentences(self, text: str):
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    def _tokenize(self, sentence: str):
        return re.findall(r"[A-Za-z0-9']+", sentence.lower())

    def fit(self, text: str):
        sentences = self._split_sentences(text)
        tokenized_sentences = []
        raw_counter = Counter()

        # Tokenize & count raw frequencies
        for s in sentences:
            tokens = self._tokenize(s)
            if tokens:
                tokenized_sentences.append(tokens)
                raw_counter.update(tokens)

        # Build vocab with UNK handling
        self.vocab = {w for w, c in raw_counter.items() if c > self.unk_threshold}
        self.vocab.update({"<UNK>", "<s>", "</s>"})

        # Reset counters
        self.trigrams = defaultdict(Counter)
        self.bigrams = defaultdict(Counter)
        self.unigrams = Counter()

        # Count n-grams
        for tokens in tokenized_sentences:
            tokens = [t if t in self.vocab else "<UNK>" for t in tokens]
            padded = ["<s>", "<s>"] + tokens + ["</s>"]

            # unigrams
            for word in padded:
                self.unigrams[word] += 1

            # bigrams
            for i in range(len(padded) - 1):
                w1, w2 = padded[i], padded[i + 1]
                self.bigrams[w1][w2] += 1

            # trigrams
            for i in range(len(padded) - 2):
                w1, w2, w3 = padded[i], padded[i + 1], padded[i + 2]
                self.trigrams[(w1, w2)][w3] += 1

        self.total_unigrams = sum(self.unigrams.values())

    def _sample_from_counter(self, counter: Counter):
        if not counter:
            return None
        words = list(counter.keys())
        weights = [counter[w] for w in words]
        return random.choices(words, weights=weights, k=1)[0]

    def generate(self, max_length=50):
        w1, w2 = "<s>", "<s>"
        generated = []

        for _ in range(max_length):
            next_word = None

            # try trigram continuation
            trigram_counter = self.trigrams.get((w1, w2))
            if trigram_counter:
                next_word = self._sample_from_counter(trigram_counter)
            else:
                # backoff → bigram
                bigram_counter = self.bigrams.get(w2)
                if bigram_counter:
                    next_word = self._sample_from_counter(bigram_counter)
                else:
                    # fallback → unigram (remove <s> so loop doesn't repeat)
                    uni = Counter(self.unigrams)
                    uni.pop("<s>", None)
                    next_word = self._sample_from_counter(uni)

            # stop if stuck
            if not next_word:
                break

            # stop sentence if end token predicted
            if next_word == "</s>":
                break

            generated.append(next_word)

            w1, w2 = w2, next_word

        return " ".join(generated)

    def fit_from_file(self, path, encoding="utf-8"):
        with open(path, "r", encoding=encoding) as f:
            text = f.read()
        self.fit(text)


if __name__ == "__main__":
    sample = """
    Alice was beginning to get very tired of sitting by her sister on the bank,
    and of having nothing to do: once or twice she had peeped into the book her
    sister was reading, but it had no pictures or conversations in it, 'and what
    is the use of a book,' thought Alice 'without pictures or conversation?'
    """

    m = TrigramModel(unk_threshold=0)
    m.fit(sample)
    print(m.generate(30))
