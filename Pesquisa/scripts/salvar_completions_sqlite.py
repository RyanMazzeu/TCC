import requests
import json
import sqlite3
from datetime import datetime

# O IP do seu notebook Windows
url = "http://192.168.1.2:11434/api/generate"

# Um prompt temático para testar
payload = {
    "model": "llama3",
    "prompt": "Escreva um parágrafo curto sobre as vantagens da arquitetura RAG para que eu e o Afonso possamos usar como base na introdução do TCC.",
    "stream": False
}

headers = {
    "Content-Type": "application/json"
}

DB_PATH = "completions.db"


def criar_tabela(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS completions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model TEXT,
            prompt TEXT,
            created_at TEXT,
            response TEXT,
            done INTEGER,
            done_reason TEXT,
            context TEXT,
            total_duration INTEGER,
            load_duration INTEGER,
            prompt_eval_count INTEGER,
            prompt_eval_duration INTEGER,
            eval_count INTEGER,
            eval_duration INTEGER,
            raw_json TEXT
        )
    """)
    conn.commit()


def salvar_resposta(conn, prompt, resposta_json):
    conn.execute("""
        INSERT INTO completions (
            timestamp, model, prompt, created_at, response, done, done_reason,
            context, total_duration, load_duration, prompt_eval_count,
            prompt_eval_duration, eval_count, eval_duration, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        resposta_json.get("model"),
        prompt,
        resposta_json.get("created_at"),
        resposta_json.get("response"),
        int(resposta_json.get("done", False)),
        resposta_json.get("done_reason"),
        json.dumps(resposta_json.get("context")),
        resposta_json.get("total_duration"),
        resposta_json.get("load_duration"),
        resposta_json.get("prompt_eval_count"),
        resposta_json.get("prompt_eval_duration"),
        resposta_json.get("eval_count"),
        resposta_json.get("eval_duration"),
        json.dumps(resposta_json),
    ))
    conn.commit()


print("Enviando requisição para o servidor no Windows... Aguarde.")

try:
    response = requests.post(url, data=json.dumps(payload), headers=headers)

    if response.status_code == 200:
        resposta_json = response.json()
        print("\n--- Resposta do Llama 3 ---")
        print(resposta_json.get("response", ""))
        print("---------------------------")

        conn = sqlite3.connect(DB_PATH)
        criar_tabela(conn)
        salvar_resposta(conn, payload["prompt"], resposta_json)
        conn.close()

        print(f"Dados salvos em '{DB_PATH}' (tabela 'completions').")
    else:
        print(f"Erro na requisição. Código de status: {response.status_code}")
        print(response.text)

except requests.exceptions.RequestException as e:
    print(f"Erro de conexão: {e}")
