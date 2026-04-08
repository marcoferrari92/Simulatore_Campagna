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

# Aggiorniamo la lista delle colonne includendo i voti singoli
cols_da_mostrare = [
    "Azienda", 
    "Score Finale", 
    "Score AI", 
    "Affinità %",
    "v_desc", 
    "v_geo", 
    "v_dip", 
    "v_fat", 
    "v_ateco", 
    "Settore", 
    "Motivo"
]

# Creiamo una mappatura per rinominare le colonne tecniche in nomi leggibili nella tabella
column_names_it = {
    "v_desc": "Voto Desc.",
    "v_geo": "Voto Geo.",
    "v_dip": "Voto Dip.",
    "v_fat": "Voto Fatt.",
    "v_ateco": "Voto ATECO"
}

if not res_df.empty:
    # 1. Assicuriamoci che i voti siano numerici
    voti_cols = ["v_desc", "v_geo", "v_dip", "v_fat", "v_ateco"]
    for col in voti_cols:
        res_df[col] = pd.to_numeric(res_df[col], errors='coerce').fillna(0)

    # 2. Calcolo Score AI (basato sui 5 pesi definiti in sidebar)
    res_df["Score AI"] = (
        (res_df["v_desc"] * wa1) + 
        (res_df["v_geo"] * wa2) + 
        (res_df["v_dip"] * wa3) + 
        (res_df["v_fat"] * wa4) + 
        (res_df["v_ateco"] * wa5)
    ).round(1)
    
    res_df["Affinità %"] = (pd.to_numeric(res_df["Sim_Raw"], errors='coerce').fillna(0) * 100).round(1)
    res_df["Score Finale"] = (res_df["Score AI"] * weight_ai) + (res_df["Affinità %"] * weight_sim)
    res_df["Score Finale"] = res_df["Score Finale"].round(1)
    
    res_df = res_df.sort_values(by="Score Finale", ascending=False)
    # Filtriamo e rinominiamo le colonne per la tabella
    display_df = res_df[cols_da_mostrare].rename(columns=column_names_it)
else:
    # Se vuoto, creiamo le intestazioni rinominate
    display_df = pd.DataFrame(columns=[column_names_it.get(c, c) for c in cols_da_mostrare])

# --- 7. VISUALIZZAZIONE RISULTATI ---
st.divider()
st.subheader("🏆 Classifica Lead Intelligente")

if not res_df.empty:
    # Applichiamo lo stile con precisione a 1 decimale e gradiente sullo Score Finale
    styled_df = display_df.style.background_gradient(
        subset=['Score Finale'], 
        cmap='YlGn'
    ).format(precision=1)
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # --- 2. MOSTRA IL GRAFICO ---
    st.divider()
    st.subheader("📊 Analisi Comparativa delle Curve di Densità")

    # 1. Aggiungiamo 'Score Finale' direttamente al mapping
    column_mapping = {
        "v_desc": "Descrizione",
        "v_geo": "Geografia",
        "v_dip": "Dipendenti",
        "v_fat": "Fatturato",
        "v_ateco": "Settore",
        "Score Finale": "TOTAL SCORE"  # <--- Lo aggiungiamo qui
    }
    
    # 2. Prepariamo i dati (ora includono anche il totale)
    df_plot = res_df[list(column_mapping.keys())].rename(columns=column_mapping)
    df_melted = df_plot.melt(var_name='Parametro', value_name='Voto')
    df_melted["Voto"] = pd.to_numeric(df_melted["Voto"], errors='coerce').fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))

    # 3. Disegniamo tutto in un colpo solo
    sns.kdeplot(
        data=df_melted, 
        x="Voto", 
        hue="Parametro", 
        fill=True,         
        common_norm=False,  
        palette="bright", 
        alpha=0.15,         
        linewidth=2,     
        ax=ax
    )

    # --- AGGIUNGI DA QUI ---
    # Recuperiamo tutte le linee create nel grafico
    lines = ax.get_lines()

    if len(lines) > 0:
        # Seaborn solitamente mette l'ultima categoria (TOTAL SCORE) in posizione 0 
        # a causa dell'ordine di sovrapposizione (z-order) predefinito.
        final_line = lines[0] 
        
        final_line.set_linestyle("--")
        final_line.set_linewidth(8)  # Ora sarà molto evidente
        final_line.set_color("black")
        final_line.set_zorder(10)    # La porta sopra tutte le altre aree colorate
    # --- A QUI ---

    # Estetica finale
    ax.set_xlim(0, 100)
    ax.set_title("Profilo di Distribuzione: Parametri vs Risultato Finale", fontsize=16)
    ax.set_xlabel("Voto (0-100)")
    ax.set_ylabel("Densità")
    ax.grid(axis='both', linestyle='--', alpha=0.3)

    st.pyplot(fig)
    
else:
    # Questa parte viene eseguita solo all'avvio (tabella vuota)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.info("In attesa di dati... Carica un file e clicca su 'Avvia Analisi'.")
