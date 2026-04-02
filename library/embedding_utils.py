from .nlp_utils import clean_text_for_embedding

def get_embedding(text, verbose=False):
    """
    Genera un embedding OpenAI filtrando prima il testo.
    Se verbose=True, stampa il testo prima e dopo il filtro.
    """
    clean_text = clean_text_for_embedding(text)

    """ Check """
    if verbose:
        print(f"--- TEXT CLEANING ---")
        print(f"ORIGINAL: {text[:100]}...")
        print(f"FILTERED: {clean_text[:100]}...")
        print(f"--------------------\n")

    response = client.embeddings.create(model="text-embedding-3-large", input=clean_text)
    return response.data[0].embedding
