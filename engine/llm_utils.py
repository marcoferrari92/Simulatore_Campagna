def valuta_llm_pro(
    client, 
    campaign, 
    company_name, 
    p1_name, p1_value,
    p2_name, p2_value,
    p3_name, p3_value,
    AI_role, AI_task, evaluation_criteria,
    max_words=15, temperature=0
):
    # Creiamo chiavi univoche per il parsing basate sui nomi dei parametri
    key1 = p1_name.upper().replace(" ", "_")
    key2 = p2_name.upper().replace(" ", "_")
    key3 = p3_name.upper().replace(" ", "_")

    prompt = f"""
    {AI_task}

    CAMPAGNA DA ANALIZZARE:
    {campaign}

    DATI AZIENDA ({company_name}):
    - {p1_name}: {p1_value}
    - {p2_name}: {p2_value}
    - {p3_name}: {p3_value}

    CRITERI DI VALUTAZIONE AI:
    {evaluation_criteria}

    Rispondi RIGOROSAMENTE in questo formato:
    VOTO_{key1}: [voto da 0 a 100]
    VOTO_{key2}: [voto da 0 a 100]
    VOTO_{key3}: [voto da 0 a 100]
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
    voti = {"v1": 0, "v2": 0, "v3": 0}
    motivo = ""

    for line in text.split("\n"):
        line_up = line.upper()
        if f"VOTO_{key1}" in line_up:
            voti["v1"] = int(''.join(filter(str.isdigit, line.split(":")[1])))
        elif f"VOTO_{key2}" in line_up:
            voti["v2"] = int(''.join(filter(str.isdigit, line.split(":")[1])))
        elif f"VOTO_{key3}" in line_up:
            voti["v3"] = int(''.join(filter(str.isdigit, line.split(":")[1])))
        elif "MOTIVO" in line_up:
            motivo = line.split(":", 1)[1].strip()

    return voti, motivo
