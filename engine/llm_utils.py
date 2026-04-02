def valuta_llm_pro(
    client, 
    campaign, 
    company_data,  # Riceve il dizionario completo della riga JSON
    p1_name, p2_name, p3_name,
    AI_role, AI_task, evaluation_criteria,
    max_words=15, temperature=0
):
    # Estraiamo i valori in base alle etichette scelte nella sidebar
    # Se la chiave non esiste nel JSON, usiamo "N/D"
    val1 = company_data.get(p1_name, "Dato non presente")
    val2 = company_data.get(p2_name, "Dato non presente")
    val3 = company_data.get(p3_name, "Dato non presente")
    company_name = company_data.get("nome", "Azienda Anonima")

    key1 = p1_name.upper().replace(" ", "_")
    key2 = p2_name.upper().replace(" ", "_")
    key3 = p3_name.upper().replace(" ", "_")

    prompt = f"""
    {AI_task}

    CAMPAGNA:
    {campaign}

    SCHEDA AZIENDA ({company_name}):
    - {p1_name}: {val1}
    - {p2_name}: {val2}
    - {p3_name}: {val3}

    CRITERI:
    {evaluation_criteria}

    Rispondi RIGOROSAMENTE in questo formato:
    VOTO_{key1}: [0-100]
    VOTO_{key2}: [0-100]
    VOTO_{key3}: [0-100]
    MOTIVO: [max {max_words} parole]
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
    voti = {"v1": 0, "v2": 0, "v3": 0}
    motivo = ""

    for line in text.split("\n"):
        line_up = line.upper()
        try:
            if f"VOTO_{key1}" in line_up:
                voti["v1"] = int(''.join(filter(str.isdigit, line.split(":")[1])))
            elif f"VOTO_{key2}" in line_up:
                voti["v2"] = int(''.join(filter(str.isdigit, line.split(":")[1])))
            elif f"VOTO_{key3}" in line_up:
                voti["v3"] = int(''.join(filter(str.isdigit, line.split(":")[1])))
            elif "MOTIVO" in line_up:
                motivo = line.split(":", 1)[1].strip()
        except: pass

    return voti, motivo
