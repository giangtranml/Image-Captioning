import config
import pickle

with open("vocab.pkl", "rb") as f:
	VOCAB = pickle.load(f)

def tokens_to_words(list_tokens):
    words = [VOCAB[t] for t in list_tokens]
    return " ".join(words)