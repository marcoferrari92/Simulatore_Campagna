def valuta_llm_pro(
    client, 
    campaign, 
    company_data,  # Dizionario con tutti i campi (nome, descrizione, regione, etc.)
    AI_role, 
    AI_task, 
    evaluation_criteria,
    max_words=50, 
    temperature=0
):
    """
    Versione Pro: Valuta l'azienda su 4 dimensioni specifiche basate sui nuovi parametri.
    """
    
    # Estrazione dati dal dizionario (con fallback N/A)
    name = company_data.get('nome', 'Azienda Anonima')
    desc = company_data.get('descrizione', 'N/D')
    loc = f"{company_data.get('citta', 'N/D')} ({company_data.get('provincia', 'N/D')}), {company_data.get('regione', 'N/D')}"
    dim = f"{company_data.get('dipendenti', 'N/D')} dipendenti, Fatturato: {company_data.get('fatturato', 'N/D')}"
    ateco = f"{company_data.get('macrosettore_ateco', 'N/D')} (Codice: {company_data.get('codice_ateco', 'N/D')})"

    prompt = f"""
    {AI_task}

    CAMPAGNA DA VALUTARE:
    {campaign}

    SCHEDA DETTAGLIATA AZIENDA:
    - NOME: {name}
    - ATTIVITÀ: {desc}
    - LOCALIZZAZIONE: {loc}
    - DIMENSIONE: {dim}
    - SETTORE ATECO: {ateco}

    CRITERI DI VALUTAZIONE AGGIUNTIVI:
    {evaluation_criteria}

    Rispondi RIGOROSAMENTE in questo formato (usa numeri interi per i voti):
    VOTO_DESCRIZIONE: [0-100]
    VOTO_GEOGRAFIA: [0-100]
    VOTO_DIMENSIONE: [0-100]
    VOTO_SETTORE: [0-100]
    MOTIVO: [spiegazione tecnica max {max_words} parole]
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": AI_role},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )

    text = response.choices[0].message.content
    
    # Inizializziamo i voti per le 4 aree
    voti = {
        "v_desc": 0,
        "v_geo": 0,
        "v_dim": 0,
        "v_ateco": 0
    }
    motivo = ""

    # Parsing delle linee
    for line in text.split("\n"):
        line_up = line.upper()
        if ":" in line:
            try:
                valore_testo = line.split(":", 1)[1]
                # Estrazione numeri per i voti
                if "VOTO_DESCRIZIONE" in line_up:
                    voti["v_desc"] = int(''.join(filter(str.isdigit, valore_testo)))
                elif "VOTO_GEOGRAFIA" in line_up:
                    voti["v_geo"] = int(''.join(filter(str.isdigit, valore_testo)))
                elif "VOTO_DIMENSIONE" in line_up:
                    voti["v_dim"] = int(''.join(filter(str.isdigit, valore_testo)))
                elif "VOTO_SETTORE" in line_up:
                    voti["v_ateco"] = int(''.join(filter(str.isdigit, valore_testo)))
                elif "MOTIVO" in line_up:
                    motivo = valore_testo.strip()
            except (ValueError, IndexError):
                continue

    return voti, motivo
