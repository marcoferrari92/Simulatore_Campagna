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

# --- SIDEBAR: Configurazione API e Accesso ---
st.sidebar.header("🔑 Accesso")
api_key = st.sidebar.text_input("Inserisci OpenAI API Key", type="password", help="Necessaria per embedding e analisi LLM")

st.sidebar.divider()

# --- SIDEBAR: Configurazione Parametri Business ---
st.sidebar.header("⚖️ Pesi e Parametri Business")

# Parametro 1 (Default: Geolocalizzazione)
p1_label = st.sidebar.text_input("Etichetta Parametro 1", "Geolocalizzazione")
p1_weight = st.sidebar.slider(f"Peso {p1_label}", 0.0, 1.0, 0.2)

# Parametro 2 (Default: Dimensione Azienda)
p2_label = st.sidebar.text_input("Etichetta Parametro 2", "Dimensione Azienda")
p2_weight = st.sidebar.slider(f"Peso {p2_label}", 0.0, 1.0, 0.2)

# Peso della Descrizione (Tecnico)
desc_weight = st.sidebar.slider("Peso Descrizione/Settore", 0.0, 1.0, 0.6)

# Normalizzazione pesi
total_w = p1_weight + p2_weight + desc_weight
if total_w > 0:
    w1, w2, wd = (p1_weight/total_w, p2_weight/total_w, desc_weight/total_w)
else:
    w1, w2, wd = (0.33, 0.33, 0.34)

st.sidebar.divider()

# --- SIDEBAR: Impostazioni AI ---
st.sidebar.header("🤖 Impostazioni AI")
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
st.title("🏢 Business Campaign Matcher Pro")
st.markdown(f"""
**Istruzioni:** Carica un file `.txt` dove ogni riga segue il formato:
`Nome Azienda : Descrizione Attività : {p1_label} : {p2_label}`
""")

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Carica il database (.txt)", type=["txt"])
with col2:
    campaign_text = st.text_area("Testo della Campagna:", 
                                 placeholder="Incolla qui il testo della newsletter o dell'offerta...",
                                 height=150)

# --- ESECUZIONE ---
if st.button("🚀 Avvia Analisi Strategica"):
    if not api_key:
        st.error("Per favore, inserisci la tua OpenAI API Key nella sidebar.")
    elif not uploaded_file or not campaign_text:
        st.warning("Assicurati di aver caricato un file e inserito il testo della campagna.")
    else:
        client = OpenAI(api_key=api_key)
        
        # Lettura file (4 Colonne)
        content = uploaded_file.read().decode('utf-8-sig', errors='ignore')
        data = []
        for line in content.splitlines():
            parti = line.split(":", 3)
            if len(parti) == 4:
                data.append({
                    "Nome": parti[0].strip(),
                    "Descrizione": parti[1].strip(),
                    "P1_val": parti[2].strip(),
                    "P2_val": parti[3].strip()
                })
        
        df = pd.DataFrame(data)
        
        if not df.empty:
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            
            # Embedding Campagna
            news_vec = emb.get_embedding(client, campaign_text)
            
            for i, row in df.iterrows():
                status_text.text(f"Analisi in corso: {row['Nome']}...")
                
                # Embedding Azienda (Semantica)
                azi_vec = emb.get_embedding(client, row['Descrizione'])
                sim = cosine_similarity([news_vec], [azi_vec])[0][0]
                
                # Chiamata LLM PRO (Assicurati che la funzione in engine accetti questi parametri)
                voti, motivo = llm.valuta_llm_pro(
                    client=client,
                    campaign=campaign_text,
                    company_name=row['Nome'],
                    company_description=row['Descrizione'],
                    param1_name=p1_label,
                    param1_value=row['P1_val'],
                    param2_name=p2_label,
                    param2_value=row['P2_val'],
                    AI_role=ai_role,
                    AI_task=ai_task,
                    evaluation_criteria=eval_criteria,
                    max_words=max_words,
                    temperature=temp
                )
                
                # Calcolo Punteggio Pesato
                score_finale = (voti["desc"] * wd) + (voti["p1"] * w1) + (voti["p2"] * w2)
                
                results.append({
                    "Azienda": row['Nome'],
                    "Score Finale": round(score_finale, 1),
                    "Analisi": motivo,
                    f"Match {p1_label}": voti["p1"],
                    f"Match {p2_label}": voti["p2"],
                    "Match Tecnico": voti["desc"],
                    "Affinità Semantica (%)": round(sim * 100, 1)
                })
                progress_bar.progress((i + 1) / len(df))
            
            status_text.success("✅ Analisi Completata!")
            
            # Risultati
            res_df = pd.DataFrame(results).sort_values(by="Score Finale", ascending=False)
            st.subheader("🏆 Ranking Lead Strategici")
            st.dataframe(res_df, use_container_width=True)
            
            # Grafico riassuntivo
            st.divider()
            st.subheader("📊 Distribuzione Top 10 Lead")
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            sns.barplot(x="Score Finale", y="Azienda", data=res_df.head(10), palette="viridis", ax=ax1)
            ax1.set_xlabel("Punteggio Composto")
            st.pyplot(fig1)
            
        else:
            st.error("Il file non contiene dati validi o non rispetta il formato 'Nome : Descrizione : P1 : P2'.")
