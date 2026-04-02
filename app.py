# Importazione funzioni
import engine.embedding_utils as emb
import engine.llm_evaluation as llm


import streamlit as st
import pandas as pd
import spacy
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


# Configurazione Pagina
st.set_page_config(page_title="Simulatore campagna marketing", layout="wide")

# --- SIDEBAR: Configurazione ---
st.sidebar.header("Configurazione")
api_key = st.sidebar.text_input("Inserisci OpenAI API Key", type="password")

# Nuovi Parametri AI (con valori di default)
st.sidebar.subheader("Impostazioni AI")
with st.sidebar.expander("Personalizza Comportamento AI", expanded=False):
    ai_role = st.text_area("AI Role", 
                           value="Sei un consulente aziendale esperto. Valuta l'interesse pratico.",
                           help="Definisce la personalità dell'AI")
    
    ai_task = st.text_area("AI Task", 
                           value="Analizza se questa azienda è interessata a ricevere questa campagna.",
                           help="L'istruzione principale per l'analisi")
    
    eval_criteria = st.text_area("Criteri di Valutazione", 
                                 value="- L'argomento è critico per l'operatività?\n- L'azienda deve applicare queste normative?",
                                 help="I punti specifici da controllare")
    
    max_words = st.number_input("Max parole (Motivo)", min_value=5, max_value=100, value=15)
    temp = st.slider("Temperature (Creatività)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

# --- UI PRINCIPALE ---
st.title("🏢 Business Campaign Matcher")
st.markdown("Carica il database aziendale e definisci i parametri della campagna per trovare i lead migliori.")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Carica il file TXT (Formato Nome : Descrizione)", type=["txt"])

with col2:
    # Cambiato da newsletter_text a campaign_text per coerenza
    campaign_text = st.text_area("Oggetto/Testo della Campagna:", 
                                 placeholder="Esempio: Nuove tecnologie per l'industria 4.0...")

# --- ESECUZIONE ---
if st.button("Avvia Analisi Strategica") and uploaded_file and campaign_text and api_key:
    client = OpenAI(api_key=api_key)
    
    # Lettura file
    content = uploaded_file.read().decode('utf-8-sig', errors='ignore')
    data = []
    for line in content.splitlines():
        if ":" in line:
            parti = line.split(":", 1)
            data.append({"Nome": parti[0].strip(), "Descrizione": parti[1].strip()})
    
    df = pd.DataFrame(data)
    
    if not df.empty:
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
        
        # Embedding Campagna
        news_vec = emb.get_embedding(client, campaign_text)
        
        for i, row in df.iterrows():
            status_text.text(f"Analisi in corso: {row['Nome']}...")
            
            # Calcolo Embedding
            azi_vec = emb.get_embedding(client, row['Descrizione'])
            sim = cosine_similarity([news_vec], [azi_vec])[0][0]
            
            # Chiamata alla nuova funzione LLM con i parametri dell'interfaccia
            score, motivo = llm.valuta_llm(
                client,
                campaign=campaign_text,
                company_name=row['Nome'],
                company_description=row['Descrizione'],
                AI_role=ai_role,
                AI_task=ai_task,
                evaluation_criteria=eval_criteria,
                max_words=max_words,
                temperature=temp
            )
            
            results.append({
                "Azienda": row['Nome'],
                "Interesse": score,
                "Analisi": motivo,
                "Affinità": round(sim * 100, 1)
            })
            progress_bar.progress((i + 1) / len(df))
        
        status_text.success("Analisi Completata!")
        
        # Risultati
        res_df = pd.DataFrame(results).sort_values(by="Interesse", ascending=False)
        
        st.subheader("🏆 Risultati Ranking")
        st.dataframe(res_df, use_container_width=True)
        
        # ... resto del codice per i grafici ...
