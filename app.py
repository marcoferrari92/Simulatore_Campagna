import json 
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import engine.embedding_utils as emb
import engine.llm_utils as llm 

# --- CONFIGURAZIONE SESSION STATE ---
if "raw_results" not in st.session_state:
    st.session_state.raw_results = None

# --- SIDEBAR: Configurazione Parametri (Pesi per l'Agente AI) ---
st.sidebar.header("⚖️ 1. Pesi Interni AI")
st.sidebar.caption("Definisci l'importanza dei dati per il giudizio dell'agente.")
w_desc = st.sidebar.slider("Peso Descrizione", 0.0, 1.0, 0.4)
w_geo = st.sidebar.slider("Peso Geografia", 0.0, 1.0, 0.1)
w_dim = st.sidebar.slider("Peso Dimensione", 0.0, 1.0, 0.2)
w_ateco = st.sidebar.slider("Peso Settore (ATECO)", 0.0, 1.0, 0.3)

# Normalizzazione pesi AI
total_ai_w = w_desc + w_geo + w_dim + w_ateco
if total_ai_w > 0:
    wa1, wa2, wa3, wa4 = w_desc/total_ai_w, w_geo/total_ai_w, w_dim/total_ai_w, w_ateco/total_ai_w
else:
    wa1, wa2, wa3, wa4 = 0.25, 0.25, 0.25, 0.25

st.sidebar.divider()

# --- SIDEBAR: Bilanciamento Algoritmo (Post-Analisi) ---
st.sidebar.header("🎛️ 2. Mix Finale")
weight_ai = st.sidebar.slider("Peso Giudizio Globale AI", 0.0, 1.0, 0.7, step=0.05)
weight_sim = 1.0 - weight_ai
st.sidebar.info(f"Mix: AI {int(weight_ai*100)}% / Similarità {int(weight_sim*100)}%")

# --- SIDEBAR: Impostazioni Agent ---
st.sidebar.divider()
st.sidebar.header("🤖 Configurazione Agente")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

with st.sidebar.expander("Parametri Avanzati", expanded=False):
    ai_role = st.text_area("AI Role", value="Sei un consulente aziendale esperto.")
    ai_task = st.text_area("AI Task", value="Valuta la compatibilità tra la campagna e il profilo aziendale.")
    eval_criteria = st.text_area("Criteri", value="- Coerenza settoriale\n- Rilevanza geografica")
    max_words = st.number_input("Max parole motivo", value=30)
    temp = st.slider("Creatività", 0.0, 1.0, 0.0)

# --- UI PRINCIPALE ---
st.title("🎯 Analizzatore Strategico Campagne")

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Carica Database (.json)", type=["json"])
with col2:
    campaign_text = st.text_area("Testo della Campagna:", placeholder="Incolla qui...", height=150)

# --- PULSANTE ESECUZIONE ---
if st.button("🚀 Avvia Analisi Multidimensionale"):
    if not uploaded_file or not campaign_text or not api_key:
        st.error("Dati mancanti o API Key assente!")
    else:
        client = OpenAI(api_key=api_key)
        try:
            data = json.load(uploaded_file)
            df = pd.DataFrame(data if isinstance(data, list) else [data])
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            temp_results = []
            
            # Embedding campagna
            news_vec = emb.get_embedding(client, campaign_text)
            
            for i, row in df.iterrows():
                status_text.text(f"Analisi AI Pro: {row.get('nome')}...")
                
                # 1. Similarità Vettoriale
                desc_azi = str(row.get('descrizione', ''))
                azi_vec = emb.get_embedding(client, desc_azi)
                sim = cosine_similarity([news_vec], [azi_vec])[0][0]
                
                # 2. Chiamata LLM PRO (Voti separati)
                voti_raw, motivo = llm.valuta_llm_pro(
                    client=client, 
                    campaign=campaign_text,
                    company_data=row.to_dict(),
                    AI_role=ai_role, 
                    AI_task=ai_task,
                    evaluation_criteria=eval_criteria,
                    max_words=max_words, 
                    temperature=temp
                )
                
                temp_results.append({
                    "Azienda": row.get('nome', 'N/A'),
                    "Settore": row.get('macrosettore_ateco', 'N/A'),
                    "Città": f"{row.get('citta')} ({row.get('provincia')})",
                    "v_desc": voti_raw["v_desc"],
                    "v_geo": voti_raw["v_geo"],
                    "v_dim": voti_raw["v_dim"],
                    "v_ateco": voti_raw["v_ateco"],
                    "Sim_Raw": sim,
                    "Analisi Strategica": motivo
                })
                progress_bar.progress((i + 1) / len(df))
            
            st.session_state.raw_results = temp_results
            status_text.success("✅ Analisi completata!")
            
        except Exception as e:
            st.error(f"Errore durante l'analisi: {e}")

# --- LOGICA DI VISUALIZZAZIONE DINAMICA ---
if st.session_state.raw_results:
    res_df = pd.DataFrame(st.session_state.raw_results)
    
    # Ricalcolo Score AI basato sui pesi della sidebar (wa1, wa2...)
    res_df["Score AI"] = (
        (res_df["v_desc"] * wa1) + 
        (res_df["v_geo"] * wa2) + 
        (res_df["v_dim"] * wa3) + 
        (res_df["v_ateco"] * wa4)
    ).round(1)
    
    # Ricalcolo Score Finale (Mix tra Score AI e Similarità)
    res_df["Affinità %"] = (res_df["Sim_Raw"] * 100).round(1)
    res_df["Score Finale"] = (res_df["Score AI"] * weight_ai) + (res_df["Affinità %"] * weight_sim)
    res_df["Score Finale"] = res_df["Score Finale"].round(1)
    
    res_df = res_df.sort_values(by="Score Finale", ascending=False)

    st.divider()
    st.subheader("🏆 Classifica Lead Intelligente")
    st.caption("Regola i pesi nella barra laterale per aggiornare il ranking istantaneamente.")
    
    # Visualizzazione Tabella
    view_cols = ["Azienda", "Score Finale", "Score AI", "Affinità %", "Settore", "Analisi Strategica"]
    styled_df = res_df[view_cols].style.background_gradient(subset=['Score Finale'], cmap='YlGn')
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Dettaglio Voti AI (Expander opzionale)
    with st.expander("🔍 Vedi dettaglio punteggi tecnici AI"):
        detail_cols = ["Azienda", "v_desc", "v_geo", "v_dim", "v_ateco"]
        st.table(res_df[detail_cols].head(10))

    # Grafico
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x="Score Finale", y="Azienda", data=res_df.head(10), palette="viridis", ax=ax)
    ax.set_title("Top 10 Lead per Compatibilità")
    st.pyplot(fig)
