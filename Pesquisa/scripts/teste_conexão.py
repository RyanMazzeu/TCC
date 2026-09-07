import requests
import json

# O IP do seu notebook Windows
url = "http://192.168.1.2:11435/api/generate"

# Um prompt temático para testar
payload = {
    "num_ctx": 2048,
    "model": "llama3.2",
    "prompt": "Escreva um parágrafo curto sobre as vantagens da arquitetura RAG para que eu e o Afonso possamos usar como base na introdução do TCC.",
    "stream": False
}

headers = {
    "Content-Type": "application/json"
}

print("Enviando requisição para o servidor no Windows... Aguarde.")

try:
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    # Verifica se a requisição foi bem sucedida (Código 200)
    if response.status_code == 200:
        resposta_json = response.json()
        print("\n--- Resposta do Llama 3 ---")
        print(resposta_json.get("response", ""))
        print("---------------------------")
    else:
        print(f"Erro na requisição. Código de status: {response.status_code}")
        print(response.text)

except requests.exceptions.RequestException as e:
    print(f"Erro de conexão: {e}")