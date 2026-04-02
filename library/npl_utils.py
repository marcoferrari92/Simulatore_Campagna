import spacy

def load_nlp():
    try:
        return spacy.load("it_core_news_sm")
    except OSError:
        spacy.cli.download("it_core_news_sm")
        return spacy.load("it_core_news_sm")

nlp = load_nlp()


# *************************
# CLEAN TEXT FOR EMBEDDING
# *************************
"""
Estrae solo le parole chiave dal testo, rimuovendo:
     t.is_stop     -> stop word: articoli, preposizioni, pronomi
     t.is_punct     -> punteggiatura
     t.is_space     -> spazi
"""
def clean_text_for_embedding(text):
    doc = nlp(str(text).lower())
    tokens = [t.text for t in doc if not t.is_stop and not t.is_punct and not t.is_space]
    return " ".join(tokens) if tokens else str(text)
