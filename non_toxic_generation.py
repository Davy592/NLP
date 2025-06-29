import pandas as pd
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
import time
import re
import random
from tqdm import tqdm

# --- 1. Configurazione dell'API di Gemini ---

# Assicurati che la tua API Key sia configurata
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("La variabile d'ambiente GOOGLE_API_KEY non è stata impostata.")
    genai.configure(api_key=api_key)
    print("API Key di Gemini configurata con successo.")
except ValueError as e:
    print(f"Attenzione: {e}. Inserisci la tua API key qui sotto.")
    API_KEY = 'INCOLLA_LA_TUA_API_KEY_QUI'
    if API_KEY == 'INCOLLA_LA_TUA_API_KEY_QUI':
        print("ERRORE: Devi sostituire 'INCOLLA_LA_TUA_API_KEY_QUI' con la tua vera chiave API.")
        exit()
    genai.configure(api_key=API_KEY)

# --- 2. Definizione dei Parametri e del Prompt ---

# Parametri per massimizzare la creatività
generation_config = {
    "temperature": 1.8, # Aumentiamo la temperatura per una maggiore creatività
    "top_p": 1.0, # Lasciamo che il modello consideri un'ampia gamma di token
    "top_k": 0,   # Non limitiamo per top_k
    "response_mime_type": "text/plain"
}

# Impostazioni di sicurezza rigorose per bloccare contenuti inappropriati
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# Inizializza il modello generativo con le impostazioni di sicurezza
model = genai.GenerativeModel(
    model_name='gemini-2.0-flash-lite',
    generation_config=generation_config,
    safety_settings=safety_settings
)

# Categorie di ruoli sani per guidare la generazione
HEALTHY_ROLES = {
    'Insicuro e Supportivo': ('Insicuro', 'Supportivo'),
    'Propositivo e Collaborativo': ('Propositivo', 'Collaborativo'),
    'Vulnerabile e Accogliente': ('Vulnerabile', 'Accogliente'),
    'Pentito e Comprensivo': ('Pentito', 'Comprensivo'),
    'Grato e Apprezzante': ('Grato', 'Apprezzante')
}

def create_generation_prompt(display_name, role1, role2):
    """Crea un prompt semplificato per generare nomi e dialogo."""
    return f"""
**Compito:**
Genera un esempio di conversazione SANA tra due partner in una coppia, basata sui ruoli assegnati.

**Dinamica da Rappresentare:**
{display_name}

**Istruzioni:**
1.  **Inventa due nomi** per i partner (un uomo e una donna, italiani).
2.  **Assegna i ruoli:** un partner sarà "{role1}", l'altro "{role2}".
3.  **Crea un dialogo** realistico e costruttivo tra loro (6-10 battute) che rifletta questa dinamica.

**Formato di Risposta Obbligatorio (due parti separate da '|||'):**

NOMI: [Nome 1], [Nome 2]
|||
DIALOGO:
[Scrivi qui il dialogo. Ogni battuta su una nuova riga, iniziando con il nome del parlante seguito da due punti.]
"""

def parse_gemini_response(response_text):
    """Estrae nomi e dialogo dalla risposta di Gemini."""
    try:
        parts = response_text.split('|||')
        if len(parts) != 2:
            return None # Formato non valido

        # Estrazione nomi
        names_part = parts[0].replace('NOMI:', '').strip()
        name1, name2 = [name.strip() for name in names_part.split(',')]

        # Estrazione e formattazione dialogo
        dialogue_part = parts[1].replace('DIALOGO:', '').strip()
        dialogue_lines = [line.strip() for line in dialogue_part.split('\n') if line.strip()]
        messages = [re.sub(r'^[A-Za-z\s]+:\s*', '', line) for line in dialogue_lines]
        final_dialogue = ' '.join([f'"{msg.strip()}"' for msg in messages if msg and len(msg.strip()) > 1])

        if not final_dialogue: return None

        return {
            'name1': name1.title(),
            'name2': name2.title(),
            'conversation': final_dialogue,
        }
    except Exception as e:
        print(f"Errore durante il parsing della risposta: {e}\nRisposta: {response_text}")
        return None


# --- 3. Processo di Generazione del Dataset ---

NUM_SAMPLES_TO_GENERATE = 950
output_filename = "Datasets/generated_healthy_conversations.csv"
final_columns = ['person_couple', 'name1', 'name2', 'conversation']

# Assicurati che la directory di output esista
output_dir = os.path.dirname(output_filename)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"\n--- Avvio generazione di {NUM_SAMPLES_TO_GENERATE} dialoghi (salvataggio riga per riga su '{output_filename}') ---")

pbar = tqdm(range(NUM_SAMPLES_TO_GENERATE), desc="Generando dialoghi sani")
generated_count = 0
for i in pbar:
    display_name, (role1, role2) = random.choice(list(HEALTHY_ROLES.items()))

    prompt = create_generation_prompt(display_name, role1, role2)
    
    try:
        # Chiamata all'API con i nuovi parametri
        response = model.generate_content(prompt)
        
        parsed_data = parse_gemini_response(response.text)
        
        if parsed_data:
            parsed_data['person_couple'] = display_name
            
            # Crea un DataFrame di una riga
            df_row = pd.DataFrame([parsed_data])
            df_row = df_row[final_columns] # Assicura l'ordine corretto delle colonne

            # Determina se scrivere l'header (solo se il file non esiste)
            header = not os.path.exists(output_filename)
            
            # Salva la riga in modalità 'append'
            df_row.to_csv(output_filename, mode='a', header=header, index=False, encoding='utf-8')
            
            generated_count += 1
            pbar.set_postfix_str(f"Salvato dialogo #{generated_count}")
        else:
            pbar.set_postfix_str("Risposta non valida, la salto.")

    except Exception as e:
        # Gestisce errori comuni, come il blocco per motivi di sicurezza
        if 'response was blocked' in str(e):
             pbar.set_postfix_str("Bloccato per sicurezza, la salto.")
        else:
            print(f"Riga {i+1}: Errore durante la chiamata API: {e}")

    time.sleep(1.5)

# --- 4. Riepilogo Finale ---

print("\n--- Generazione completata ---")

if generated_count > 0:
    print(f"Aggiunti {generated_count} nuovi dialoghi al file: '{output_filename}'")
    print("Esecuzione terminata con successo.")
else:
    print("Nessun dato è stato generato in questa sessione. Controlla eventuali errori API o di sicurezza.")