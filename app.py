import json 
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import engine.embedding_utils as emb
import engine.llm_utils as llm # Assicurati che qui ci sia la nuova funzione valuta_llm

# --- SIDEBAR: Configurazione Parametri ---
st.sidebar.header("⚖️ Pesi Valutazione Strategica")

# Definiamo i parametri fissi che hai richiesto
st.sidebar.subheader("Pesi Criteri")
w_desc = st.sidebar.slider("Peso Descrizione Aziendale", 0.0, 1.0, 0.4)
w_geo = st.sidebar.slider("Peso Localizzazione (Regione/Prov/Città)", 0.0, 1.0, 0.1)
w_dim = st.sidebar.slider("Peso Dimensione (Dipendenti/Fatturato)", 0.0, 1.0, 0.2)
w_ateco = st.sidebar.slider("Peso Settore (ATECO)", 0.0, 1.0, 0.3)

# Normalizzazione pesi
total_w = w_desc + w_geo + w_dim + w_ateco
if total_w > 0:
    w1, w2, w3, w4 = w_desc/total_w, w_geo/total_w, w_dim/total_w, w_ateco/total_w
else:
    w1, w2, w3, w4 = 0.25, 0.25, 0.25, 0.25

st.sidebar.divider()

# --- SIDEBAR: Impostazioni Agent ---
st.sidebar.header("🤖 Impostazioni Agent")
api_key = st.sidebar.text_input("Inserisci OpenAI API Key", type="password")

with st.sidebar.expander("Personalizza Comportamento AI", expanded=False):
    ai_role = st.text_area("AI Role", value="Sei un consulente aziendale esperto che lavora dentro l'azienda target.")
    ai_task = st.text_area("AI Task", value="Valuta la compatibilità tra la campagna e il profilo aziendale.")
    eval_criteria = st.text_area("Criteri di Valutazione", value="- Coerenza settoriale\n- Rilevanza geografica\n- Capacità finanziaria/dimensionale")
    max_words = st.number_input("Max parole (Motivo)", min_value=5, max_value=100, value=25)
    temp = st.slider("Creatività (temp.)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

# --- UI PRINCIPALE ---
st.title("🎯 Target Discovery AI")

# Box informativo con elenco puntato
st.info(" Segui questi passaggi per configurare l'analisi:")
st.markdown(f"""
1. **Chiave API**: Inserisci la tua **OpenAI API Key** nel campo dedicato in basso nella barra laterale.
2. **Database Clienti**: Carica il file `.json` contenente l'elenco delle aziende.
3. **Campagna**: Incolla il testo della tua **campagna marketing** o newsletter nel riquadro a destra.
4. **Parametri e Pesi**: Definisci i nomi dei parametri e regola la loro importanza tramite gli slider.
5. **AI Agent**: Se necessario, espandi la sezione "Personalizza Comportamento AI" per modificare il ruolo o i criteri di valutazione dell'agente.
""")

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Carica JSON Aziende", type=["json"])
    st.caption("Il JSON deve contenere: nome, descrizione, regione, provincia, citta, dipendenti, fatturato, macrosettore_ateco, codice_ateco")
with col2:
    campaign_text = st.text_area("Testo Campagna:", placeholder="Incolla qui la newsletter...", height=150)

# --- ESECUZIONE ---
if st.button("🚀 Avvia Analisi Multidimensionale") and uploaded_file and campaign_text and api_key:
    client = OpenAI(api_key=api_key)
    
    try:
        data = json.load(uploaded_file)
        df = pd.DataFrame(data if isinstance(data, list) else [data])
    except Exception as e:
        st.error(f"Errore JSON: {e}")
        df = pd.DataFrame()

    if not df.empty:
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
        
        # Embedding della campagna per calcolo affinità semantica base
        news_vec = emb.get_embedding(client, campaign_text)
        
        for i, row in df.iterrows():
            status_text.text(f"Analisi avanzata: {row.get('nome', 'Azienda')}...")
            
            # 1. Calcolo Affinità Semantica (solo su descrizione)
            desc_azi = str(row.get('descrizione', ''))
            azi_vec = emb.get_embedding(client, desc_azi)
            sim = cosine_similarity([news_vec], [azi_vec])[0][0]
            
            # 2. Chiamata alla nuova funzione LLM con i nuovi parametri
            # Nota: i nomi delle chiavi nel .get() devono corrispondere al tuo JSON
            score_ai, motivo = llm.valuta_llm(
                client=client,
                campaign=campaign_text,
                company_name=row.get('nome', 'N/A'),
                company_description=desc_azi,
                AI_role=ai_role,
                AI_task=ai_task,
                evaluation_criteria=eval_criteria,
                region=row.get('regione', 'N/A'),
                province=row.get('provincia', 'N/A'),
                city=row.get('citta', 'N/A'),
                employees_count=row.get('dipendenti', 'N/A'),
                revenue=row.get('fatturato', 'N/A'),
                ateco_macro_sector=row.get('macrosettore_ateco', 'N/A'),
                ateco_code=row.get('codice_ateco', 'N/A'),
                max_words=max_words,
                temperature=temp
            )
            
            # 3. Calcolo Score Finale Combinato
            # Uniamo l'affinità semantica (vettore) con il giudizio dell'AI
            # (Puoi regolare questa formula come preferisci)
            final_score = (score_ai * 0.7) + (sim * 100 * 0.3) 
            
            results.append({
                "Azienda": row.get('nome', 'N/A'),
                "Settore": row.get('macrosettore_ateco', 'N/A'),
                "Città": f"{row.get('citta')} ({row.get('provincia')})",
                "Score AI": score_ai,
                "Affinità %": round(sim * 100, 1),
                "Score Finale": round(final_score, 1),
                "Analisi Strategica": motivo
            })
            progress_bar.progress((i + 1) / len(df))
            
        status_text.success("✅ Analisi Completata!")
        res_df = pd.DataFrame(results).sort_values(by="Score Finale", ascending=False)

        # Visualizzazione Tabella
        st.subheader("🏆 Classifica Lead Qualificati")
        styled_df = res_df.style.background_gradient(subset=['Score Finale'], cmap='YlGn')
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Grafico top 10
        st.divider()
        st.subheader("📊 Top 10 Aziende per Compatibilità")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Score Finale", y="Azienda", data=res_df.head(10), palette="magma", ax=ax)
        st.pyplot(fig)
