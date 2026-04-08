def valuta_llm(
    client,  
    campaign, 
    company_name, 
    company_description, 
    AI_role, 
    AI_task, 
    evaluation_criteria,
    region,
    province,
    city,
    employees_count,
    revenue,
    ateco_macro_sector,
    ateco_code,
    max_words=15, 
    temperature=0
):
    """
    Funzione di valutazione potenziata con dati demografici, finanziari e settoriali (ATECO).
    """
    
    prompt = f"""
    {AI_task}

    --- DATI AZIENDA ---
    NOME: {company_name}
    DESCRIZIONE: {company_description}
    SETTORE (Macro Ateco): {ateco_macro_sector}
    CODICE ATECO: {ateco_code}
    LOCALIZZAZIONE: {city} ({province}), {region}
    DIMENSIONE: {employees_count} dipendenti
    FATTURATO: {revenue}

    --- DETTAGLI CAMPAGNA ---
    ARGOMENTO/CAMPAGNA: {campaign}

    CRITERI DI VALUTAZIONE:
    {evaluation_criteria}

    Rispondi RIGOROSAMENTE in questo formato:
    COMPATIBILITA: [voto da 0 a 100]
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
    score = 0
    reason = ""

    # Parsing dei risultati migliorato
    for line in text.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            key_upper = key.strip().upper()
            
            if "COMPATIBILITA" in key_upper:
                try:
                    score = int(''.join(filter(str.isdigit, value)))
                except ValueError: 
                    score = 0
            elif "MOTIVO" in key_upper:
                reason = value.strip()

    return score, reason
