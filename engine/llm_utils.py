def valuta_llm_pro(
    client, 
    campaign, 
    company_data, 
    AI_role, 
    AI_task, 
    evaluation_criteria,
    max_words=50, 
    temperature=0
):
    """
    Versione Pro+: Valuta l'azienda su 5 dimensioni specifiche per una precisione massima.
    """
    
    # Estrazione dati granulare
    name = company_data.get('nome', 'Azienda Anonima')
    desc = company_data.get('descrizione', 'N/D')
    loc = f"{company_data.get('citta', 'N/D')} ({company_data.get('provincia', 'N/D')}), {company_data.get('regione', 'N/D')}"
    employees = company_data.get('dipendenti', 'N/D')
    revenue = company_data.get('fatturato', 'N/D')
    ateco = f"{company_data.get('macrosettore_ateco', 'N/D')} (Codice: {company_data.get('codice_ateco', 'N/D')})"

    prompt = f"""
    {AI_task}

    CAMPAGNA DA VALUTARE:
    {campaign}

    SCHEDA DETTAGLIATA AZIENDA:
    - NOME: {name}
    - ATTIVITÀ: {desc}
    - LOCALIZZAZIONE: {loc}
    - NUMERO DIPENDENTI: {employees}
    - FATTURATO ANNUO: {revenue}
    - SETTORE ATECO: {ateco}

    CRITERI DI VALUTAZIONE AGGIUNTIVI:
    {evaluation_criteria}

    Rispondi RIGOROSAMENTE in questo formato (voti da 0 a 100):
    VOTO_DESCRIZIONE: [voto]
    VOTO_GEOGRAFIA: [voto]
    VOTO_DIPENDENTI: [voto]
    VOTO_FATTURATO: [voto]
    VOTO_SETTORE: [voto]
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
    
    # Inizializziamo i 5 voti
    voti = {
        "v_desc": 0, "v_geo": 0, "v_dip": 0, "v_fat": 0, "v_ateco": 0
    }
    motivo = ""

    for line in text.split("\n"):
        line_up = line.upper()
        if ":" in line:
            try:
                valore_testo = line.split(":", 1)[1]
                if "VOTO_DESCRIZIONE" in line_up:
                    voti["v_desc"] = int(''.join(filter(str.isdigit, valore_testo)))
                elif "VOTO_GEOGRAFIA" in line_up:
                    voti["v_geo"] = int(''.join(filter(str.isdigit, valore_testo)))
                elif "VOTO_DIPENDENTI" in line_up:
                    voti["v_dip"] = int(''.join(filter(str.isdigit, valore_testo)))
                elif "VOTO_FATTURATO" in line_up:
                    voti["v_fat"] = int(''.join(filter(str.isdigit, valore_testo)))
                elif "VOTO_SETTORE" in line_up:
                    voti["v_ateco"] = int(''.join(filter(str.isdigit, valore_testo)))
                elif "MOTIVO" in line_up:
                    motivo = valore_testo.strip()
            except: continue

    return voti, motivo
