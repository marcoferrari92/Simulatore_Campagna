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

# --- SIDEBAR: Configurazione Parametri Custom ---
st.sidebar.header("⚖️ Pesi e Parametri")

# Parametro 1 (Default: Geolocalizzazione)
p1_label = st.sidebar.text_input("Etichetta Parametro 1", "Geolocalizzazione")
p1_weight = st.sidebar.slider(f"Peso {p1_label}", 0.0, 1.0, 0.2)

# Parametro 2 (Default: Dimensione Azienda)
p2_label = st.sidebar.text_input("Etichetta Parametro 2", "Dimensione Azienda")
p2_weight = st.sidebar.slider(f"Peso {p2_label}", 0.0, 1.0, 0.2)

# Peso della Descrizione (Tecnico)
desc_weight = st.sidebar.slider("Peso Descrizione/Settore", 0.0, 1.0, 0.6)

# Normalizzazione pesi per calcolo matematico
total_w = p1_weight + p2_weight + desc_weight
if total_w > 0:
    w1, w2, wd = p1_weight/total_w, p2_weight/total_w, desc_weight/total_w
else:
    w1, w2, wd = 0.33, 0.33, 0.34

# --- UI PRINCIPALE ---
st.title("🏢 Business Campaign Matcher Pro")
st.info(f"Formato file richiesto: `Nome : Descrizione : {p1_label} : {p2_label}`")

# ... (codice per uploaded_file e campaign_text rimane uguale) ...

# --- ESECUZIONE ---
if st.button("Avvia Analisi Strategica") and uploaded_file and campaign_text and api_key:
    client = OpenAI(api_key=api_key)
    
    # 1. LETTURA FILE MIGLIORATA (4 Colonne)
    content = uploaded_file.read().decode('utf-8-sig', errors='ignore')
    data = []
    for line in content.splitlines():
        parti = line.split(":", 3) # Dividiamo in max 4 parti
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
        
        # Embedding Campagna (per l'Affinità semantica generale)
        news_vec = emb.get_embedding(client, campaign_text)
        
        for i, row in df.iterrows():
            status_text.text(f"Analisi in corso: {row['Nome']}...")
            
            # Calcolo Embedding per confronto semantico
            azi_vec = emb.get_embedding(client, row['Descrizione'])
            sim = cosine_similarity([news_vec], [azi_vec])[0][0]
            
            # 2. CHIAMATA ALLA FUNZIONE PRO (Ritorna un dizionario di voti)
            voti, motivo = llm.valuta_llm_pro(
                client,
                campaign=campaign_text,
                company_name=row['Nome'],
                company_description=row['Descrizione'],
                param1_name=p1_label,
                param1_value=row['P1_val'],
                param2_name=p2_label,
                param2_value=row['P2_val'],
                AI_role=ai_role,
                AI_task=ai_task,
                max_words=max_words,
                temperature=temp
            )
            
            # 3. CALCOLO PUNTEGGIO PESATO
            # final_score = (VotoDesc * PesoDesc) + (VotoP1 * PesoP1) + (VotoP2 * PesoP2)
            score_finale = (voti["desc"] * wd) + (voti["p1"] * w1) + (voti["p2"] * w2)
            
            results.append({
                "Azienda": row['Nome'],
                "Score Finale": round(score_finale, 1),
                "Analisi": motivo,
                f"Match {p1_label}": voti["p1"],
                f"Match {p2_label}": voti["p2"],
                "Match Tecnico": voti["desc"],
                "Affinità Semantica": round(sim * 100, 1)
            })
            progress_bar.progress((i + 1) / len(df))
        
        status_text.success("Analisi Completata!")
        
        # Risultati ordinati per lo Score Finale (quello pesato)
        res_df = pd.DataFrame(results).sort_values(by="Score Finale", ascending=False)
        
        st.subheader("🏆 Risultati Ranking Strategico")
        st.dataframe(res_df, use_container_width=True)
        
        # --- GRAFICI ---
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            fig1, ax1 = plt.subplots()
            sns.barplot(x="Score Finale", y="Azienda", data=res_df.head(10), palette="viridis", ax=ax1)
            ax1.set_title("Top 10 Lead (Punteggio Pesato)")
            st.pyplot(fig1)
        with c2:
            fig2, ax2 = plt.subplots()
            # Confronto tra Match Tecnico e Match Geografico/P1
            sns.scatterplot(x=f"Match {p1_label}", y="Match Tecnico", size="Score Finale", data=res_df, ax=ax2)
            ax2.set_title(f"Distribuzione {p1_label} vs Tecnico")
            st.pyplot(fig2)
