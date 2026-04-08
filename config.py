# config.py

# Parametri di default
Giudizio_AI = 1.0
Max_Words = 30
Creativity = 0.0

# Configurazione Agente Default
AI_ROLE_DEFAULT = "Sei un consulente aziendale esperto dell'azienda target."
AI_TASK_DEFAULT = "Valuta la compatibilità tra la campagna e il profilo aziendale."
EVAL_CRITERIA_DEFAULT = "- Coerenza settoriale\n- Rilevanza geografica\n- Capacità finanziaria/dimensionale"



# Titoli
Sidebar_AI_Title = "Modifica Comportamento AI (Richiede nuovo avvio)"
Sidebar_AI_Role_Title = "AI Role"
Sidebar_AI_Task_Title = "AI Task"
Sidebar_AI_Criteria_Title = "AI Criteria"
Sidebar_AI_Answer_Title = "Answer max words"

# Testi di default
TXT_DEFAULT_CAMPAIGN = """Newsletter che spiega nuovi DPI indispensabili per lo stoccaggio di materiali pericolosi o tossici, quali prodotti da reazioni chimiche, per un corso di formazione organizzato a Vicenza."""

# Testi informativi UI
HELP_PESI = "💡Come funziona il calcolo?"
HELP_CALCOLO_PESI = """
    **Logica di Valutazione:**
    1. L'agente AI analizza ogni azienda del database e assegna un **punteggio indipendente** a ciascun parametro.
    2. Questi voti vengono salvati nel sistema e usati a fine analisi per calcolare lo **score finale** per ogni azienda.
    3. Dopo l'analisi del database, puoi comunque modificare nuovamente i pesi in **tempo reale**: lo score finale viene ricalcolato istantaneamente senza nuove chiamate API e analisi del database.
  """

HELP_Istruzioni = "💡Istruzioni"
HELP_Generale = """
1. **Chiave API**: Inserisci la tua **OpenAI API Key** nel campo dedicato nella barra laterale.
2. **Database Clienti**: Carica il file `.json` del tuo database aziende.
3. **Campagna**: Incolla il testo della tua **campagna marketing** o newsletter nel riquadro a destra.
4. **Parametri e Pesi**: Definisci i parametri e regola la loro importanza tramite gli slider.
5. **AI e Similarità**: Puoi regolare anche quanto è predominante, nel calcolo dello score finale, l'analisi AI rispetto alla Similarità (questa definisce l'affinità tra le parole chiave del testo della campagna con quelle della descrizione di ogni azienda)"
5. **AI Agent**: Se necessario, espandi la sezione "Modifica Comportamento AI" per modificare il ruolo, il task o i criteri di valutazione dell'agente AI.
"""

WARNING_TAB = "Muovi gli slider dei pesi per ricalcolare la classifica istantaneamente senza analizzare nuovamente il database"
WARNING_CREDITS = "⚠️ **Attenzione:** L'avvio dell'analisi comporterà l'invio di dati alle API di OpenAI e il relativo consumo di crediti (Token)."

