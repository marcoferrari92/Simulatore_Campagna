import spacy

def load_nlp():
    try:
        return spacy.load("it_core_news_sm")
    except OSError:
        spacy.cli.download("it_core_news_sm")
        return spacy.load("it_core_news_sm")

nlp = load_nlp()

def clean_text_for_embedding(text):
    """Estrae solo le parole chiave dal testo."""
    doc = nlp(str(text).lower())
    tokens = [t.text for t in doc if not t.is_stop and not t.is_punct and not t.is_space]
    return " ".join(tokens) if tokens else str(text)
