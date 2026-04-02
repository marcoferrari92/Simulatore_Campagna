from engine.nlp_utils import clean_text_for_embedding

def get_embedding(client, text):
    """Pulisce il testo e richiama l'embedding di OpenAI."""
    processed_text = clean_text_for_embedding(text)
    
    response = client.embeddings.create(
        model="text-embedding-3-large", 
        input=processed_text
    )
    
    return response.data[0].embedding, processed_text
