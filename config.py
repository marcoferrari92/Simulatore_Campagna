# config.py

# Testi di default
DEFAULT_CAMPAIGN = """Siamo leader nella fornitura di soluzioni software per l'automazione dei processi industriali e il monitoraggio energetico tramite AI."""

# Testi informativi UI
HELP_CALCOLO_PESI = """
    **Logica di Valutazione:**
    1. L'agente AI analizza ogni azienda del database e assegna un **punteggio indipendente** a ciascun parametro.
    2. Questi voti vengono salvati nel sistema e usati a fine analisi per calcolare lo **score finale** per ogni azienda.
    3. Dopo l'analisi del database, puoi comunque modificare nuovamente i pesi in **tempo reale**: lo score finale viene ricalcolato istantaneamente senza nuove chiamate API e analisi del database.
  """

WARNING_CREDITS = "⚠️ **Attenzione:** L'avvio dell'analisi comporterà l'invio di dati alle API di OpenAI e il relativo consumo di crediti (Token)."

# Configurazione Agente Default
AI_ROLE_DEFAULT = "Sei un consulente aziendale esperto."
AI_TASK_DEFAULT = "Valuta la compatibilità tra la campagna e il profilo aziendale."
EVAL_CRITERIA_DEFAULT = "- Coerenza settoriale\n- Rilevanza geografica\n- Capacità finanziaria/dimensionale"
