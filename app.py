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
if "raw_results" not in st.session_state:
    # Inizializziamo con una lista vuota invece di None per far funzionare la tabella da subito
    st.session_state.raw_results = []

# --- 2. SIDEBAR: IMPOSTAZIONI AGENTE (Richiedono nuova analisi) ---
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
with st.sidebar.expander("Modifica Comportamento AI (Richiede nuovo avvio)"):
    ai_role = st.text_area("AI Role", value="Sei un consulente esperto.")
    ai_task = st.text_area("AI Task", value="Valuta la compatibilità.")
    eval_criteria = st.text_area("Criteri", value="- Coerenza core business\n- Target dimensionale")
    max_words = st.number_input("Max parole", value=30)
    temp = st.slider("Creatività", 0.0, 1.0, 0.0)

# --- 3. SIDEBAR: PESI E PARAMETRI (Ricalcolo Istantaneo) ---
st.sidebar.divider()
st.sidebar.header("🎛️ Bilanciamento Dinamico")

# Pesi per il mix finale (AI vs Similarità)
st.sidebar.divider()
st.sidebar.subheader("Mix Finale")
weight_ai = st.sidebar.slider("Peso Giudizio Globale AI", 0.0, 1.0, 0.7, step=0.05)
weight_sim = 1.0 - weight_ai

# Pesi per comporre lo Score AI (wa1-wa4)
st.sidebar.subheader("Pesi Interni Agente")
w_desc = st.sidebar.slider("Descrizione", 0.0, 1.0, 0.4)
w_geo = st.sidebar.slider("Geografia", 0.0, 1.0, 0.1)
w_dim = st.sidebar.slider("Dimensione", 0.0, 1.0, 0.2)
w_ateco = st.sidebar.slider("Settore (ATECO)", 0.0, 1.0, 0.3)

# Normalizzazione automatica dei pesi interni
total_ai_w = w_desc + w_geo + w_dim + w_ateco
wa1, wa2, wa3, wa4 = (w_desc/total_ai_w, w_geo/total_ai_w, w_dim/total_ai_w, w_ateco/total_ai_w) if total_ai_w > 0 else (0.25, 0.25, 0.25, 0.25)



# --- 4. UI PRINCIPALE ---
st.title("Simulatore campagna marketing")

# Box informativo con elenco puntato
st.info(" Segui questi passaggi per configurare l'analisi:")
st.markdown(f"""
1. **Chiave API**: Inserisci la tua **OpenAI API Key** nel campo dedicato in basso nella barra laterale.
2. **Database Clienti**: Carica il file `.json` contenente l'elenco delle aziende.
3. **Campagna**: Incolla il testo della tua **campagna marketing** o newsletter nel riquadro a destra.
4. **Parametri e Pesi**: Definisci i parametri e regola la loro importanza tramite gli slider.
5. **AI Agent**: Se necessario, espandi la sezione "Personalizza Comportamento AI" per modificare il ruolo, il task o i criteri di valutazione dell'agente AI.
""")

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

# Messaggio di attenzione sotto il tasto
st.warning("⚠️ **Attenzione:** L'avvio dell'analisi comporterà l'invio di dati alle API di OpenAI e il relativo consumo di crediti (Token).")

# --- 6. LOGICA DI CALCOLO DINAMICO ---
# Creiamo il DataFrame partendo dallo stato (che sarà vuoto all'inizio o pieno dopo l'analisi)
res_df = pd.DataFrame(st.session_state.raw_results)

# Verifichiamo se il DF ha dati per evitare errori di calcolo su colonne inesistenti
if not res_df.empty:
    # Ricalcolo Score AI
    res_df["Score AI"] = (
        (res_df["v_desc"] * wa1) + 
        (res_df["v_geo"] * wa2) + 
        (res_df["v_dim"] * wa3) + 
        (res_df["v_ateco"] * wa4)
    ).round(1)
    
    # Calcolo finale
    res_df["Affinità %"] = (res_df["Sim_Raw"] * 100).round(1)
    res_df["Score Finale"] = (res_df["Score AI"] * weight_ai) + (res_df["Affinità %"] * weight_sim)
    res_df["Score Finale"] = res_df["Score Finale"].round(1)
    res_df = res_df.sort_values(by="Score Finale", ascending=False)
else:
    # Se vuoto, creiamo le colonne di visualizzazione vuote per non far rompere la tabella
    cols_vuote = ["Azienda", "Score Finale", "Score AI", "Affinità %", "Settore", "Motivo"]
    res_df = pd.DataFrame(columns=cols_vuote)

# --- 7. VISUALIZZAZIONE RISULTATI (Sempre visibile) ---
st.divider()
st.subheader("🏆 Classifica Lead Intelligente")
st.caption("Muovi gli slider dei pesi per ricalcolare la classifica istantaneamente senza analizzare nuovamente il database")
    
# La tabella sarà sempre visibile. Se res_df è vuoto, mostrerà solo le intestazioni.
st.dataframe(
    res_df.style.background_gradient(subset=['Score Finale'], cmap='YlGn') if not res_df.empty else res_df,
    use_container_width=True, 
    hide_index=True
)
