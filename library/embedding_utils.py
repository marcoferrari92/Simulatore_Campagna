from engine.nlp_utils import clean_text_for_embedding

def get_embedding(client, text):
    """Pulisce il testo e richiama l'embedding di OpenAI."""
    processed_text = clean_text_for_embedding(text)
    
    response = client.embeddings.create(
        model="text-embedding-3-large", 
        input=processed_text
    )
    
    return response.data[0].embedding, processed_text


def get_embedding(text, verbose=False):
    """
    Genera un embedding OpenAI filtrando prima il testo.
    Se verbose=True, stampa il testo prima e dopo il filtro.
    """
    doc = nlp(str(text).lower())

    tokens_puliti = [
        token.text for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]

    clean_text = " ".join(tokens_puliti) if tokens_puliti else str(text)

    if verbose:
        print(f"--- TEXT CLEANING ---")
        print(f"ORIGINAL: {text[:100]}...")
        print(f"FILTERED: {clean_text[:100]}...")
        print(f"--------------------\n")

    response = client.embeddings.create(model="text-embedding-3-large", input=clean_text)
    return response.data[0].embedding
