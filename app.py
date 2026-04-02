import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import engine.embedding_utils as emb
import engine.llm_evaluation as llm

# --- SIDEBAR: Configurazione Parametri ---
st.sidebar.header("⚖️ Pesi e Parametri Campagna")

# Parametro 1 (Es. Descrizione)
p1_label = st.sidebar.text_input("Parametro 1", "Descrizione")
p1_weight = st.sidebar.slider(f"Peso {p1_label}", 0.0, 1.0, 0.6)

# Parametro 2 (Es. Geolocalizzazione)
p2_label = st.sidebar.text_input("Parametro 2", "Geolocalizzazione")
p2_weight = st.sidebar.slider(f"Peso {p2_label}", 0.0, 1.0, 0.2)

# Parametro 3 (Es. Dimensione Azienda)
p3_label = st.sidebar.text_input("Parametro 3", "Dimensione Azienda")
p3_weight = st.sidebar.slider(f"Peso {p3_label}", 0.0, 1.0, 0.2)

# Normalizzazione pesi
total_w = p1_weight + p2_weight + p3_weight
if total_w > 0:
    w1 = p1_weight / total_w
    w2 = p2_weight / total_w
    w3 = p3_weight / total_w
else:
    w1, w2, w3 = 0.33, 0.33, 0.34

st.sidebar.divider()

# --- SIDEBAR: Impostazioni Agent ---
st.sidebar.header("🤖 Impostazioni Agent")
api_key = st.sidebar.text_input("Inserisci OpenAI API Key", type="password")

with st.sidebar.expander("Personalizza Comportamento AI", expanded=False):
    ai_role = st.text_area("AI Role", value="Sei un consulente aziendale esperto. Valuta l'interesse pratico.")
    ai_task = st.text_area("AI Task", value="Analizza se questa azienda è interessata a ricevere questa campagna.")
    eval_criteria = st.text_area("Criteri di Valutazione", value="- L'argomento è critico per l'operatività?\n- L'azienda deve applicare queste normative?")
    max_words = st.number_input("Max parole (Motivo)", min_value=5, max_value=100, value=15)
    temp = st.slider("Temperature (Creatività)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

# --- UI PRINCIPALE ---
st.title("🏢 Analizzatore campagna marketing")
st.markdown(f"**Formato file richiesto:** `Nome Azienda : {p1_label} : {p2_label} : {p3_label}`")

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Carica il database (.txt)", type=["txt"])
with col2:
    campaign_text = st.text_area("Testo della Campagna:", placeholder="Incolla qui la newsletter...", height=150)

# --- ESECUZIONE ---
if st.button("🚀 Avvia Analisi Strategica"):
    if not api_key:
        st.error("Inserisci la API Key.")
    elif not uploaded_file or not campaign_text:
        st.warning("Carica un file e inserisci il testo della campagna.")
    else:
        client = OpenAI(api_key=api_key)
        content = uploaded_file.read().decode('utf-8-sig', errors='ignore')
        data = []
        
        for line in content.splitlines():
            parti = line.split(":", 3)
            if len(parti) == 4:
                data.append({
                    "Nome": parti[0].strip(),
                    "P1_val": parti[1].strip(),
                    "P2_val": parti[2].strip(),
                    "P3_val": parti[3].strip()
                })
        
        df = pd.DataFrame(data)
        
        if not df.empty:
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            
            # Embedding Campagna (Usiamo P1_val come riferimento semantico principale per l'affinità)
            news_vec = emb.get_embedding(client, campaign_text)
            
            for i, row in df.iterrows():
                status_text.text(f"Analisi: {row['Nome']}...")
                
                # Embedding Affinità (rispetto al Parametro 1, solitamente la descrizione)
                azi_vec = emb.get_embedding(client, row['P1_val'])
                sim = cosine_similarity([news_vec], [azi_vec])[0][0]
                
                # Chiamata LLM PRO
                voti, motivo = llm.valuta_llm_pro(
                    client=client,
                    campaign=campaign_text,
                    company_name=row['Nome'],
                    p1_name=p1_label, p1_value=row['P1_val'],
                    p2_name=p2_label, p2_value=row['P2_val'],
                    p3_name=p3_label, p3_value=row['P3_val'],
                    AI_role=ai_role,
                    AI_task=ai_task,
                    evaluation_criteria=eval_criteria,
                    max_words=max_words,
                    temperature=temp
                )
                
                # Calcolo Finale Pesato
                score_finale = (voti["v1"] * w1) + (voti["v2"] * w2) + (voti["v3"] * w3)
                
                results.append({
                    "Azienda": row['Nome'],
                    "Score Finale": round(score_finale, 1),
                    "Analisi": motivo,
                    f"Match {p1_label}": voti["v1"],
                    f"Match {p2_label}": voti["v2"],
                    f"Match {p3_label}": voti["v3"],
                    "Affinità Semantica (%)": round(sim * 100, 1)
                })
                progress_bar.progress((i + 1) / len(df))
            
            status_text.success("✅ Analisi Completata!")
            res_df = pd.DataFrame(results).sort_values(by="Score Finale", ascending=False)
            st.subheader("🏆 Ranking Lead Strategici")
            st.dataframe(res_df, use_container_width=True)
            
            # Grafico
            st.divider()
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            sns.barplot(x="Score Finale", y="Azienda", data=res_df.head(10), palette="viridis", ax=ax1)
            st.pyplot(fig1)
