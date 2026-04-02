import streamlit as st
import pandas as pd
import spacy
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Iportazione funzioni
import library.embedding_utils as emb

# Configurazione Pagina
st.set_page_config(page_title="Analizzatore Newsletter", layout="wide")

# --- SIDEBAR: Configurazione ---
st.sidebar.header("Configurazione")
api_key = st.sidebar.text_input("Inserisci OpenAI API Key", type="password")

# --- UI PRINCIPALE ---
st.title("🏢 Business Newsletter Matcher")
st.markdown("Carica il database aziendale e incolla il testo della newsletter per trovare i lead migliori.")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Carica il file TXT (Formato Nome : Descrizione)", type=["txt"])

with col2:
    newsletter_text = st.text_area("Descrizione della newsletter:", placeholder="Esempio: Nuove normative sulla sicurezza sul lavoro...")

# --- LOGICA CORE ---
def valuta_llm(client, newsletter, azienda, descrizione):
    prompt = f"NEWSLETTER: {newsletter}\nAZIENDA: {azienda}\nATTIVITÀ: {descrizione}\nValuta compatibilità 0-100 e motivo breve."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Rispondi solo in formato COMPATIBILITA: [voto]\nMOTIVO: [max 15 parole]"},
                  {"role": "user", "content": prompt}],
        temperature=0
    )
    res = response.choices[0].message.content
    # Parsing veloce
    score = 0
    motivo = "N/D"
    for line in res.split("\n"):
        if "COMPATIBILITA" in line.upper():
            score = int(''.join(filter(str.isdigit, line)))
        if "MOTIVO" in line.upper():
            motivo = line.split(":", 1)[1].strip()
    return score, motivo

# --- ESECUZIONE ---
if st.button("Avvia Analisi Strategica") and uploaded_file and newsletter_text and api_key:
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
        results = []
        
        # Embedding Newsletter
        news_vec = emb.get_embedding(client, newsletter_text)
        
        for i, row in df.iterrows():
            # Calcolo Embedding e LLM
            azi_vec = emb.get_embedding(client, row['Descrizione'])
            sim = cosine_similarity([news_vec], [azi_vec])[0][0]
            score, motivo = valuta_llm(client, newsletter_text, row['Nome'], row['Descrizione'])
            
            results.append({
                "Azienda": row['Nome'],
                "Interesse": score,
                "Analisi": motivo,
                "Affinità": round(sim * 100, 1)
            })
            progress_bar.progress((i + 1) / len(df))
        
        # Risultati
        res_df = pd.DataFrame(results).sort_values(by="Interesse", ascending=False)
        
        st.subheader("🏆 Risultati Ranking")
        st.dataframe(res_df, use_container_width=True)
        
        # Grafici
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            fig1, ax1 = plt.subplots()
            sns.histplot(res_df["Interesse"], color="skyblue", kde=True, ax=ax1)
            ax1.set_title("Distribuzione Interesse (LLM)")
            st.pyplot(fig1)
        with c2:
            fig2, ax2 = plt.subplots()
            sns.histplot(res_df["Affinità"], color="salmon", kde=True, ax=ax2)
            ax2.set_title("Distribuzione Affinità (Embedding)")
            st.pyplot(fig2)
    else:
        st.error("Il file caricato non sembra essere nel formato corretto.")
else:
    st.info("Configura l'API Key, carica un file e scrivi la descrizione per iniziare.")
