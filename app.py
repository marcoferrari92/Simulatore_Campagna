import json 
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import engine.embedding_utils as emb
import engine.llm_utils as llm 
import config

# --- 1. INIZIALIZZAZIONE SESSION STATE ---
if "raw_results" not in st.session_state:
    st.session_state.raw_results = []

# --- 2. SIDEBAR: IMPOSTAZIONI AGENTE ---
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
with st.sidebar.expander(config.Sidebar_AI_Title):
    ai_role = st.text_area(config.Sidebar_AI_Role_Title, value=config.AI_ROLE_DEFAULT)
    ai_task = st.text_area(config.Sidebar_AI_Task_Title, value=config.AI_TASK_DEFAULT)
    eval_criteria = st.text_area(config.Sidebar_AI_Criteria_Title, value=config.EVAL_CRITERIA_DEFAULT)
    max_words = st.number_input(config.Sidebar_AI_Answer_Title, value=config.Max_Words)
    temp = st.slider("Creatività", 0.0, 1.0, config.Creativity)

# --- 3. SIDEBAR: PESI E PARAMETRI (5 Parametri ora) ---
st.sidebar.divider()
st.sidebar.header("🎛️ Bilanciamento Dinamico")
with st.sidebar.popover(config.HELP_PESI):
     st.markdown(config.HELP_CALCOLO_PESI)

# Pesi per il mix finale (AI vs Similarità)
st.sidebar.subheader("Mix Finale")
weight_ai = st.sidebar.slider("Peso Giudizio Globale AI", 0.0, 1.0, config.Giudizio_AI, step=0.05)
weight_sim = 1.0 - weight_ai

# Pesi per comporre lo Score AI (5 variabili: wa1-wa5)
st.sidebar.subheader("Pesi Interni Agente")
w_desc = st.sidebar.slider("Descrizione Attività", 0.0, 1.0, 0.3)
w_geo = st.sidebar.slider("Geografia", 0.0, 1.0, 0.1)
w_dip = st.sidebar.slider("N. Dipendenti", 0.0, 1.0, 0.15)
w_fat = st.sidebar.slider("Fatturato", 0.0, 1.0, 0.15)
w_ateco = st.sidebar.slider("Settore (ATECO)", 0.0, 1.0, 0.3)

# Normalizzazione automatica dei pesi interni su 5 dimensioni
total_ai_w = w_desc + w_geo + w_dip + w_fat + w_ateco
if total_ai_w > 0:
    wa1, wa2, wa3, wa4, wa5 = w_desc/total_ai_w, w_geo/total_ai_w, w_dip/total_ai_w, w_fat/total_ai_w, w_ateco/total_ai_w
else:
    wa1, wa2, wa3, wa4, wa5 = 0.2, 0.2, 0.2, 0.2, 0.2

# --- 4. UI PRINCIPALE ---
st.title("Simulatore campagna marketing")
st.divider()
with st.popover(config.HELP_Istruzioni):
     st.markdown(config.HELP_Generale)
st.divider()

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Carica Database JSON", type=["json"])
with col2:
    campaign_text = st.text_area(
        "Testo Campagna:", 
        value=config.TXT_DEFAULT_CAMPAIGN,
        height=250
    )

# --- 5. ESECUZIONE ANALISI ---
if st.button("🚀 Esegui Analisi AI"):
    if not (uploaded_file and campaign_text and api_key):
        st.error("Compila tutti i campi prima di procedere.")
    else:
        client = OpenAI(api_key=api_key)
        data = json.load(uploaded_file)
        df = pd.DataFrame(data)
        
        results_storage = []
        progress = st.progress(0)
        status = st.empty()
        
        news_vec = emb.get_embedding(client, campaign_text)
        
        for i, row in df.iterrows():
            status.text(f"Analisi {row.get('nome')}...")
            
            desc_azi = str(row.get('descrizione', ''))
            azi_vec = emb.get_embedding(client, desc_azi)
            sim_raw = cosine_similarity([news_vec], [azi_vec])[0][0]
            
            # Valutazione AI PRO (Richiede engine aggiornato con v_dip e v_fat)
            voti_raw, motivo = llm.valuta_llm_pro(
                client=client, campaign=campaign_text, company_data=row.to_dict(),
                AI_role=ai_role, AI_task=ai_task, evaluation_criteria=eval_criteria,
                max_words=max_words, temperature=temp
            )
            
            results_storage.append({
                "Azienda": row.get('nome'),
                "Settore": row.get('macrosettore_ateco'),
                "Sim_Raw": sim_raw,
                "v_desc": voti_raw["v_desc"],
                "v_geo": voti_raw["v_geo"],
                "v_dip": voti_raw["v_dip"], # Nuovo parametro
                "v_fat": voti_raw["v_fat"], # Nuovo parametro
                "v_ateco": voti_raw["v_ateco"],
                "Motivo": motivo
            })
            progress.progress((i + 1) / len(df))
            
        st.session_state.raw_results = results_storage
        status.success("Analisi completata! Ora puoi regolare i pesi a sinistra.")

st.warning(config.WARNING_CREDITS)

# --- 6. LOGICA DI CALCOLO DINAMICO ---
res_df = pd.DataFrame(st.session_state.raw_results)

if not res_df.empty:
    # Ricalcolo Score AI basato sui 5 pesi normalizzati
    res_df["Score AI"] = (
        (res_df["v_desc"] * wa1) + 
        (res_df["v_geo"] * wa2) + 
        (res_df["v_dip"] * wa3) + 
        (res_df["v_fat"] * wa4) + 
        (res_df["v_ateco"] * wa5)
    ).round(1)
    
    # Calcolo finale
    res_df["Affinità %"] = (res_df["Sim_Raw"] * 100).round(1)
    res_df["Score Finale"] = (res_df["Score AI"] * weight_ai) + (res_df["Affinità %"] * weight_sim)
    res_df["Score Finale"] = res_df["Score Finale"].round(1)
    res_df = res_df.sort_values(by="Score Finale", ascending=False)
else:
    cols_vuote = ["Azienda", "Score Finale", "Score AI", "Affinità %", "Settore", "Motivo"]
    res_df = pd.DataFrame(columns=cols_vuote)

# --- 7. VISUALIZZAZIONE RISULTATI ---
st.divider()
st.subheader("🏆 Classifica Lead Intelligente")
st.caption(config.WARNING_TAB)
    
st.dataframe(
    res_df.style.background_gradient(subset=['Score Finale'], cmap='YlGn') if not res_df.empty else res_df,
    use_container_width=True, 
    hide_index=True
)

if not res_df.empty:
    st.subheader("📊 Top 10 Lead")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x="Score Finale", y="Azienda", data=res_df.head(10), palette="viridis", ax=ax)
    st.pyplot(fig)

if not res_df.empty:
    st.divider()
    st.subheader("📊 Distribuzione dei Punteggi AI")
    st.caption("Visualizza come i voti dell'agente si distribuiscono tra i vari parametri analizzati.")

    # Creiamo 5 colonne per i 5 istogrammi dei parametri AI
    hist_cols = st.columns(5)
    
    # Mappatura parametri per i grafici
    metrics = [
        ("Descrizione", "v_desc", "Blues"),
        ("Geografia", "v_geo", "Reds"),
        ("Dipendenti", "v_dip", "Greens"),
        ("Fatturato", "v_fat", "Oranges"),
        ("Settore", "v_ateco", "Purples")
    ]

    for col, (label, col_name, color) in zip(hist_cols, metrics):
        with col:
            fig, ax = plt.subplots(figsize=(4, 3))
            # Creiamo l'istogramma
            sns.histplot(res_df[col_name], bins=10, kde=True, color=color[:color.find('s')].lower(), ax=ax)
            
            # Pulizia estetica del grafico
            ax.set_title(label, fontsize=12)
            ax.set_xlabel("Voto", fontsize=10)
            ax.set_ylabel("Frequenza", fontsize=10)
            ax.set_xlim(0, 100) # Tutti i voti sono 0-100
            
            st.pyplot(fig)

    # Manteniamo il grafico del Ranking finale sotto
    st.divider()
    st.subheader("📊 Top 10 Lead (Score Finale)")
    fig_rank, ax_rank = plt.subplots(figsize=(10, 4))
    sns.barplot(x="Score Finale", y="Azienda", data=res_df.head(10), palette="viridis", ax=ax_rank)
    st.pyplot(fig_rank)
