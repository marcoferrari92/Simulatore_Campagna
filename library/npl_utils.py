import spacy

def load_nlp():
    try:
        return spacy.load("it_core_news_sm")
    except OSError:
        spacy.cli.download("it_core_news_sm")
        return spacy.load("it_core_news_sm")

nlp = load_nlp()
