# config.py

# Testi di default
TXT_DEFAULT_CAMPAIGN = """Newsletter che spiega nuovi DPI indispensabili per lo stoccaggio di materiali pericolosi o tossici, quali prodotti da reazioni chimiche, per un corso di formazione organizzato a Vicenza."""

# Testi informativi UI
HELP_CALCOLO_PESI = """
    **Logica di Valutazione:**
    1. L'agente AI analizza ogni azienda del database e assegna un **punteggio indipendente** a ciascun parametro.
    2. Questi voti vengono salvati nel sistema e usati a fine analisi per calcolare lo **score finale** per ogni azienda.
    3. Dopo l'analisi del database, puoi comunque modificare nuovamente i pesi in **tempo reale**: lo score finale viene ricalcolato istantaneamente senza nuove chiamate API e analisi del database.
  """

HELP_GENERALE = """
1. **Chiave API**: Inserisci la tua **OpenAI API Key** nel campo dedicato nella barra laterale.
2. **Database Clienti**: Carica il file `.json` del tuo database aziende.
3. **Campagna**: Incolla il testo della tua **campagna marketing** o newsletter nel riquadro a destra.
4. **Parametri e Pesi**: Definisci i parametri e regola la loro importanza tramite gli slider.
5. **AI Agent**: Se necessario, espandi la sezione "Modifica Comportamento AI" per modificare il ruolo, il task o i criteri di valutazione dell'agente AI.
"""

WARNING_TAB = "Muovi gli slider dei pesi per ricalcolare la classifica istantaneamente senza analizzare nuovamente il database"
WARNING_CREDITS = "⚠️ **Attenzione:** L'avvio dell'analisi comporterà l'invio di dati alle API di OpenAI e il relativo consumo di crediti (Token)."

# Configurazione Agente Default
AI_ROLE_DEFAULT = "Sei un consulente aziendale esperto."
AI_TASK_DEFAULT = "Valuta la compatibilità tra la campagna e il profilo aziendale."
EVAL_CRITERIA_DEFAULT = "- Coerenza settoriale\n- Rilevanza geografica\n- Capacità finanziaria/dimensionale"

