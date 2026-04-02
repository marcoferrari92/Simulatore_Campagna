def llm_evaluation(
    campaign, 
    company_name, 
    company_description, 
    AI_role, 
    AI_task, 
    evaluation_criteria, 
    max_words=15, 
    temperature=0
):
    """
    Funzione di valutazione con variabili in inglese e prompt in italiano.
    """
    
    prompt = f"""
    {AI_task}

    CAMPAGNA (Argomento): {campaign}
    AZIENDA (Nome): {company_name}
    ATTIVITÀ AZIENDALE: {company_description}

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

    # Parsing dei risultati
    for line in text.split("\n"):
        line_upper = line.upper()
        if "COMPATIBILITA" in line_upper:
            try:
                # Estrae il voto numerico
                score = int(''.join(filter(str.isdigit, line.split(":")[1])))
            except (IndexError, ValueError): 
                pass
        elif "MOTIVO" in line_upper:
            try:
                # Estrae la spiegazione
                reason = line.split(":", 1)[1].strip()
            except IndexError:
                pass

    return score, reason
