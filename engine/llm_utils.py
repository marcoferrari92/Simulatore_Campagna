def valuta_llm_pro(
    client, 
    campaign, 
    company_name, 
    company_description, 
    param1_name, param1_value,
    param2_name, param2_value,
    AI_role, AI_task, 
    max_words=15, temperature=0
):
    prompt = f"""
    {AI_task}

    DATI AZIENDA:
    - Nome: {company_name}
    - Descrizione: {company_description}
    - {param1_name.upper()}: {param1_value}
    - {param2_name.upper()}: {param2_value}

    DETTAGLI CAMPAGNA:
    {campaign}

    Analizza la compatibilità basandoti su tre pilastri: Descrizione Tecnica, {param1_name} e {param2_name}.
    
    Rispondi RIGOROSAMENTE in questo formato:
    VOTO_DESCRIZIONE: [0-100]
    VOTO_{param1_name.upper().replace(' ', '_')}: [0-100]
    VOTO_{param2_name.upper().replace(' ', '_')}: [0-100]
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
    
    # Parsing dinamico dei voti
    results = {"desc": 0, "p1": 0, "p2": 0}
    reason = ""
    
    p1_key = f"VOTO_{param1_name.upper().replace(' ', '_')}"
    p2_key = f"VOTO_{param2_name.upper().replace(' ', '_')}"

    for line in text.split("\n"):
        line_up = line.upper()
        if "VOTO_DESCRIZIONE" in line_up:
            results["desc"] = int(''.join(filter(str.isdigit, line.split(":")[1])))
        elif p1_key in line_up:
            results["p1"] = int(''.join(filter(str.isdigit, line.split(":")[1])))
        elif p2_key in line_up:
            results["p2"] = int(''.join(filter(str.isdigit, line.split(":")[1])))
        elif "MOTIVO" in line_up:
            reason = line.split(":", 1)[1].strip()

    return results, reason
