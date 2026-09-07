import requests
import json
import time
from datetime import datetime

def verificar_status_gpu(base_url):
    """
    Consulta a API do Ollama para descobrir matematicamente
    quantos bytes do modelo estão na GPU vs Processador (RAM).
    """
    url_ps = f"{base_url}/api/ps"
    try:
        response = requests.get(url_ps)
        if response.status_code == 200:
            modelos_carregados = response.json().get("models", [])
            
            if not modelos_carregados:
                print("Nenhum modelo carregado na memória no momento.")
                return
            
            print("\n--- Diagnóstico de Hardware (GPU) ---")
            for modelo in modelos_carregados:
                nome = modelo.get("name")
                tamanho_total = modelo.get("size", 1) # Evita divisão por zero
                tamanho_vram = modelo.get("size_vram", 0)
                detalhes = modelo.get("details", {})

                # Convertendo bytes para Megabytes (MB)
                total_mb = tamanho_total / (1024 * 1024)
                vram_mb = tamanho_vram / (1024 * 1024)

                # Calcula a porcentagem exata que está na placa de vídeo
                percentual_gpu = (tamanho_vram / tamanho_total) * 100

                print(f"Modelo na Memória: {nome}")
                print(f"Parâmetros:        {detalhes.get('parameter_size', 'N/A')}")
                print(f"Quantização:       {detalhes.get('quantization_level', 'N/A')}")
                print(f"Família:           {detalhes.get('family', 'N/A')}")
                print(f"Tamanho Total:     {total_mb:.2f} MB")
                print(f"Alocado na GPU:    {vram_mb:.2f} MB")
                print(f"Expira em:         {modelo.get('expires_at', 'N/A')}")

                if percentual_gpu >= 99.0:
                    print(f"Offload Status:    ✅ {percentual_gpu:.1f}% na GPU (Desempenho Máximo)")
                else:
                    print(f"Offload Status:    ⚠️ {percentual_gpu:.1f}% na GPU (Gargalo: Offload Parcial para RAM)")
            print("-------------------------------------")
            return modelos_carregados
    except Exception as e:
        print(f"\n⚠️ Não foi possível verificar o status da GPU: {e}")

def testar_ollama_benchmark(modelo, prompt):
    # Separamos a URL base para poder reaproveitar
    base_url = "http://192.168.1.15:11435"
    url_generate = f"{base_url}/api/generate"
    
    payload = {
        "model": modelo,
        "prompt": prompt,
        "stream": False,
        "keep_alive": -1,
        "options": {
            "temperature": 0.0, 
            "seed": 42,         
            "num_ctx": 4096     # Trava de segurança da memória de contexto
        }
    }
    
    headers = {"Content-Type": "application/json"}
    print(f"Enviando requisição para {url_generate}\nModelo: {modelo}\nProcessando... Aguarde.")
    
    try:
        inicio_req = time.time()
        response = requests.post(url_generate, data=json.dumps(payload), headers=headers)
        fim_req = time.time()
        
        if response.status_code == 200:
            dados = response.json()
            resposta_texto = dados.get("response", "")

            tokens_gerados = dados.get("eval_count", 0)
            tempo_gpu_ns = dados.get("eval_duration", 0)

            tokens_prompt = dados.get("prompt_eval_count", 0)
            tempo_prompt_ns = dados.get("prompt_eval_duration", 0)
            tempo_load_ns = dados.get("load_duration", 0)
            tempo_total_ns = dados.get("total_duration", 0)
            done_reason = dados.get("done_reason", "N/A")

            tempo_gpu_s = tempo_gpu_ns / 1e9
            tempo_prompt_s = tempo_prompt_ns / 1e9
            tempo_load_s = tempo_load_ns / 1e9
            tempo_total_s = tempo_total_ns / 1e9
            tempo_total_req = fim_req - inicio_req

            tps = 0
            if tempo_gpu_s > 0:
                tps = tokens_gerados / tempo_gpu_s

            prompt_tps = 0
            if tempo_prompt_s > 0:
                prompt_tps = tokens_prompt / tempo_prompt_s

            # Time to First Token: tempo até o primeiro token sair (carga + processamento do prompt)
            ttft_s = tempo_load_s + tempo_prompt_s

            # Overhead: parte do tempo total que não é carga, prompt eval nem geração
            overhead_s = tempo_total_s - (tempo_load_s + tempo_prompt_s + tempo_gpu_s)

            print("\n--- Resposta da IA ---")
            print(resposta_texto)
            print("\n--- Métricas de Desempenho (Benchmark) ---")
            print(f"Tempo Total de Espera:   {tempo_total_req:.2f} segundos")
            print(f"Motivo de Parada:        {done_reason}")
            print(f"Tempo de Carga do Modelo:{tempo_load_s:.2f} segundos")
            print(f"Tokens do Prompt:        {tokens_prompt} tokens")
            print(f"Tempo Proc. do Prompt:   {tempo_prompt_s:.2f} segundos ({prompt_tps:.2f} tokens/s)")
            print(f"Time to First Token:     {ttft_s:.2f} segundos")
            print(f"Tempo Real de Processam: {tempo_gpu_s:.2f} segundos na GPU")
            print(f"Tokens Gerados:          {tokens_gerados} tokens")
            print(f"Velocidade de Inferência:{tps:.2f} tokens/segundo")
            print(f"Overhead do Pipeline:    {overhead_s:.2f} segundos")
            print("------------------------------------------")

            # --- NOVIDADE: Chama a verificação da GPU logo após gerar o texto ---
            modelos_carregados = verificar_status_gpu(base_url)
            detalhes_modelo = {}
            if modelos_carregados:
                for m in modelos_carregados:
                    if m.get("name") == modelo or m.get("model") == modelo:
                        detalhes_modelo = m.get("details", {})
                        break

            log_data = {
                "timestamp": datetime.now().isoformat(),
                "modelo": modelo,
                "prompt": prompt,
                "resposta": resposta_texto,
                "done_reason": done_reason,
                "tokens_prompt": tokens_prompt,
                "tempo_prompt_s": round(tempo_prompt_s, 3),
                "prompt_tokens_por_segundo": round(prompt_tps, 2),
                "tempo_load_s": round(tempo_load_s, 3),
                "time_to_first_token_s": round(ttft_s, 3),
                "tokens_gerados": tokens_gerados,
                "tempo_processamento_s": round(tempo_gpu_s, 3),
                "tokens_por_segundo": round(tps, 2),
                "tempo_total_s": round(tempo_total_s, 3),
                "overhead_pipeline_s": round(overhead_s, 3),
                "parameter_size": detalhes_modelo.get("parameter_size", "N/A"),
                "quantization_level": detalhes_modelo.get("quantization_level", "N/A"),
                "family": detalhes_modelo.get("family", "N/A")
            }
            
            with open("benchmark_tcc_logs.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
                
            print("-> Log salvo em 'benchmark_tcc_logs.jsonl'")
            
        else:
            print(f"Erro na requisição. Código: {response.status_code}")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"Erro de conexão. Verifique se o servidor Windows está rodando na porta 11435: {e}")

if __name__ == "__main__":
    MODELO_ATUAL = "qwen2.5:1.5b" #"llama3.2" 
    PROMPT_TESTE = "Me diga uma receita de miojo."
    
    testar_ollama_benchmark(MODELO_ATUAL, PROMPT_TESTE)