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
# Inizializziamo il contenitore dei risultati se non esiste
if "raw_results" not in st.session_state:
    st.session_state.raw_results = None

# --- SIDEBAR: Bilanciamento Dinamico (Sempre accessibile) ---
st.sidebar.header("🎛️ Bilanciamento Real-Time")
weight_ai = st.sidebar.slider("Peso Giudizio Agente AI", 0.0, 1.0, 0.7, step=0.05)
weight_sim = 1.0 - weight_ai
st.sidebar.caption(f"Mix: AI {int(weight_ai*100)}% / Sim. {int(weight_sim*100)}%")

# --- SIDEBAR: Impostazioni Agent ---
st.sidebar.divider()
st.sidebar.header("🤖 Configurazione AI")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

with st.sidebar.expander("Parametri Avanzati Agente", expanded=False):
    ai_role = st.text_area("AI Role", value="Sei un consulente aziendale esperto.")
    ai_task = st.text_area("AI Task", value="Valuta la compatibilità tra la campagna e il profilo aziendale.")
    eval_criteria = st.text_area("Criteri", value="- Coerenza settoriale\n- Rilevanza geografica")
    max_words = st.number_input("Max parole motivo", value=25)
    temp = st.slider("Creatività", 0.0, 1.0, 0.0)

# --- UI PRINCIPALE ---
st.title("Analizzatore campagna marketing")

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Carica JSON Aziende", type=["json"])
with col2:
    campaign_text = st.text_area("Testo Campagna:", placeholder="Incolla qui...", height=150)

# --- PULSANTE ESECUZIONE (Chiamata API) ---
if st.button("🚀 Genera Analisi (Consuma API)"):
    if not uploaded_file or not campaign_text or not api_key:
        st.error("Inserisci tutti i dati e la API Key!")
    else:
        client = OpenAI(api_key=api_key)
        try:
            data = json.load(uploaded_file)
            df = pd.DataFrame(data if isinstance(data, list) else [data])
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            temp_results = []
            
            news_vec = emb.get_embedding(client, campaign_text)
            
            for i, row in df.iterrows():
                status_text.text(f"Analisi AI: {row.get('nome')}...")
                
                # 1. Similarità
                desc_azi = str(row.get('descrizione', ''))
                azi_vec = emb.get_embedding(client, desc_azi)
                sim = cosine_similarity([news_vec], [azi_vec])[0][0]
                
                # 2. LLM
                score_ai, motivo = llm.valuta_llm(
                    client=client, campaign=campaign_text,
                    company_name=row.get('nome', 'N/A'),
                    company_description=desc_azi,
                    AI_role=ai_role, AI_task=ai_task,
                    evaluation_criteria=eval_criteria,
                    region=row.get('regione', 'N/A'),
                    province=row.get('provincia', 'N/A'),
                    city=row.get('citta', 'N/A'),
                    employees_count=row.get('dipendenti', 'N/A'),
                    revenue=row.get('fatturato', 'N/A'),
                    ateco_macro_sector=row.get('macrosettore_ateco', 'N/A'),
                    ateco_code=row.get('codice_ateco', 'N/A'),
                    max_words=max_words, temperature=temp
                )
                
                temp_results.append({
                    "Azienda": row.get('nome', 'N/A'),
                    "Settore": row.get('macrosettore_ateco', 'N/A'),
                    "Città": f"{row.get('citta')} ({row.get('provincia')})",
                    "Score_AI_Raw": score_ai,
                    "Sim_Raw": sim,
                    "Analisi Strategica": motivo
                })
                progress_bar.progress((i + 1) / len(df))
            
            # Salviamo tutto nello stato della sessione
            st.session_state.raw_results = temp_results
            status_text.success("✅ Dati API ottenuti con successo!")
            
        except Exception as e:
            st.error(f"Errore: {e}")

# --- LOGICA DI VISUALIZZAZIONE (Ricalcolo istantaneo) ---
if st.session_state.raw_results:
    # Creiamo il DataFrame dai risultati salvati
    res_df = pd.DataFrame(st.session_state.raw_results)
    
    # Applichiamo il calcolo dei pesi in tempo reale
    res_df["Affinità %"] = (res_df["Sim_Raw"] * 100).round(1)
    res_df["Score Finale"] = (res_df["Score_AI_Raw"] * weight_ai) + (res_df["Affinità %"] * weight_sim)
    res_df["Score Finale"] = res_df["Score Finale"].round(1)
    
    # Ordiniamo per il nuovo score
    res_df = res_df.sort_values(by="Score Finale", ascending=False)

    st.divider()
    st.subheader("🏆 Classifica Lead Dinamica")
    st.caption("Muovi gli slider a sinistra per cambiare la classifica istantaneamente senza ricaricare i dati.")
    
    # Visualizzazione Tabella
    view_cols = ["Azienda", "Score Finale", "Score_AI_Raw", "Affinità %", "Settore", "Analisi Strategica"]
    styled_df = res_df[view_cols].style.background_gradient(subset=['Score Finale'], cmap='YlGn')
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Grafico top 10
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x="Score Finale", y="Azienda", data=res_df.head(10), palette="magma", ax=ax)
    st.pyplot(fig)
