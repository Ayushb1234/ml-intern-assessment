<!-- # Evaluation

Please provide a 1-page summary of your design choices for the Trigram Language Model.

This should include:

- How you chose to store the n-gram counts.
- How you handled text cleaning, padding, and unknown words.
- How you implemented the `generate` function and the probabilistic sampling.
- Any other design decisions you made and why you made them. -->


# Evaluation — Trigram Language Model (1 page)


**Storage of n-gram counts**
I store trigram counts using a mapping from a bigram context to a Counter of next words:
`trigrams[(w1,w2)] -> Counter(w3)`. This structure is compact, simple to update, and efficient for sampling
because it directly gives all continuations for a given context. I also keep `bigrams` and `unigrams` (both
Counter-based) to implement a straightforward backoff strategy when the trigram context is unseen.


**Text cleaning and tokenization**
Text is split into sentences using punctuation (`.`, `!`, `?`) and then tokenized using the regex
`[A-Za-z0-9']+` to preserve simple contractions and alphanumeric tokens. Everything is lowercased for
consistency. This keeps tokenization lightweight and deterministic without external libraries.


**Padding and sentence boundaries**
I pad each sentence with two start tokens `'<s>'` and a single end token `'</s>'`. This makes it possible to
learn sentence-beginning trigrams and to stop generation nicely when an end token is sampled.


**Unknown words**
To robustly handle rare words, I replace tokens whose training frequency is `<= unk_threshold` with a
special `<UNK>` token. The default threshold is 1 (so words seen only once are treated as unknown). This
stabilizes probabilities and keeps vocabulary size in check.


**Generation and probabilistic sampling**
Generation starts from the `('<s>','<s>')` context and repeatedly samples the next word from the
conditional distribution defined by trigram counts. Sampling uses `random.choices()` with counts as
weights, which realizes probabilistic selection (rather than greedy argmax). If the trigram context
is unseen, the model backs off to a bigram distribution (words following the current last token). If
that is also missing, it falls back to a unigram distribution (excluding `<s>` to avoid loops).


**Design tradeoffs**
- I chose a simple backoff strategy (trigram -> bigram -> unigram) to keep the model intuitive and
easy to implement. More sophisticated smoothing (Kneser-Ney, Laplace) improves performance but adds
complexity and hyperparameters; it was not required here.
- Tokenization is lightweight and not language-aware; t