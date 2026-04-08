import json 
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import engine.embedding_utils as emb
import engine.llm_utils as llm 

# --- 1. INIZIALIZZAZIONE SESSION STATE ---
# Questo serve a mantenere i dati estratti dall'AI anche quando sposti gli slider
if "raw_results" not in st.session_state:
    st.session_state.raw_results = None

# --- 2. SIDEBAR: PESI E PARAMETRI (Ricalcolo Istantaneo) ---
st.sidebar.header("🎛️ Bilanciamento Dinamico")

# Pesi per comporre lo Score AI (wa1-wa4)
st.sidebar.subheader("Pesi Interni Agente")
w_desc = st.sidebar.slider("Descrizione", 0.0, 1.0, 0.4)
w_geo = st.sidebar.slider("Geografia", 0.0, 1.0, 0.1)
w_dim = st.sidebar.slider("Dimensione", 0.0, 1.0, 0.2)
w_ateco = st.sidebar.slider("Settore (ATECO)", 0.0, 1.0, 0.3)

# Pesi per il mix finale (AI vs Similarità)
st.sidebar.divider()
st.sidebar.subheader("Mix Finale")
weight_ai = st.sidebar.slider("Peso Giudizio Globale AI", 0.0, 1.0, 0.7, step=0.05)
weight_sim = 1.0 - weight_ai

# Normalizzazione automatica dei pesi interni
total_ai_w = w_desc + w_geo + w_dim + w_ateco
wa1, wa2, wa3, wa4 = (w_desc/total_ai_w, w_geo/total_ai_w, w_dim/total_ai_w, w_ateco/total_ai_w) if total_ai_w > 0 else (0.25, 0.25, 0.25, 0.25)

# --- 3. SIDEBAR: IMPOSTAZIONI AGENTE (Richiedono nuova analisi) ---
st.sidebar.divider()
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
with st.sidebar.expander("Modifica Comportamento AI (Richiede nuovo avvio)"):
    ai_role = st.text_area("AI Role", value="Sei un consulente esperto.")
    ai_task = st.text_area("AI Task", value="Valuta la compatibilità.")
    eval_criteria = st.text_area("Criteri", value="- Coerenza core business\n- Target dimensionale")
    max_words = st.number_input("Max parole", value=30)
    temp = st.slider("Creatività", 0.0, 1.0, 0.0)

# --- 4. UI PRINCIPALE ---
st.title("🎯 Analizzatore Lead con Pesi Dinamici")

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Carica Database JSON", type=["json"])
with col2:
    # Definiamo il testo di default
default_campaign = """Newsletter che spiega nuovi DPI indispensabili per lo stoccaggio di materiali pericolosi o tossici, quali prodotti da reazioni chimiche, per un corso di formazione organizzato a Vicenza."""
    campaign_text = st.text_area(
        "Testo Campagna:", 
        value=default_campaign,
        height=250
    )

# --- 5. ESECUZIONE ANALISI (Solo se cliccato il tasto) ---
if st.button("🚀 Esegui Analisi AI (Consuma Crediti)"):
    if not (uploaded_file and campaign_text and api_key):
        st.error("Compila tutti i campi prima di procedere.")
    else:
        client = OpenAI(api_key=api_key)
        data = json.load(uploaded_file)
        df = pd.DataFrame(data)
        
        results_storage = []
        progress = st.progress(0)
        status = st.empty()
        
        # Generiamo embedding campagna una volta sola
        news_vec = emb.get_embedding(client, campaign_text)
        
        for i, row in df.iterrows():
            status.text(f"Analisi {row.get('nome')}...")
            
            # Calcolo Similarità (Grezzo)
            desc_azi = str(row.get('descrizione', ''))
            azi_vec = emb.get_embedding(client, desc_azi)
            sim_raw = cosine_similarity([news_vec], [azi_vec])[0][0]
            
            # Valutazione AI (4 voti separati)
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
                "v_dim": voti_raw["v_dim"],
                "v_ateco": voti_raw["v_ateco"],
                "Motivo": motivo
            })
            progress.progress((i + 1) / len(df))
            
        # SALVATAGGIO IN SESSION STATE
        st.session_state.raw_results = results_storage
        status.success("Analisi completata! Ora puoi regolare i pesi a sinistra.")

# --- 6. LOGICA DI CALCOLO DINAMICO (Sempre attiva se ci sono dati) ---
if st.session_state.raw_results:
    # Trasformiamo i dati salvati in un DataFrame locale
    res_df = pd.DataFrame(st.session_state.raw_results)
    
    # Ricalcoliamo lo Score AI usando i pesi della sidebar (wa1, wa2...)
    res_df["Score AI"] = (
        (res_df["v_desc"] * wa1) + 
        (res_df["v_geo"] * wa2) + 
        (res_df["v_dim"] * wa3) + 
        (res_df["v_ateco"] * wa4)
    ).round(1)
    
    # Calcolo finale (Mix tra AI e Similarità)
    res_df["Affinità %"] = (res_df["Sim_Raw"] * 100).round(1)
    res_df["Score Finale"] = (res_df["Score AI"] * weight_ai) + (res_df["Affinità %"] * weight_sim)
    res_df["Score Finale"] = res_df["Score Finale"].round(1)
    
    res_df = res_df.sort_values(by="Score Finale", ascending=False)

    # --- 7. VISUALIZZAZIONE RISULTATI ---
    st.divider()
    st.subheader("🏆 Ranking Ottimizzato")
    
    # Formattazione e visualizzazione
    display_df = res_df[["Azienda", "Score Finale", "Score AI", "Affinità %", "Settore", "Motivo"]]
    st.dataframe(
        display_df.style.background_gradient(subset=['Score Finale'], cmap='YlGn'),
        use_container_width=True, hide_index=True
    )
    
    # Grafico top 10
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x="Score Finale", y="Azienda", data=res_df.head(10), palette="viridis", ax=ax)
    st.pyplot(fig)
