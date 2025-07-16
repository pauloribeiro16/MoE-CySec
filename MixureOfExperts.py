import ollama
import json
import time

# ----- Classe Expert (sem alterações) -----
# Continua a ser a nossa definição de um especialista
class Expert:
    def __init__(self, name: str, description: str, model_name: str, system_prompt: str):
        self.name = name
        self.description = description  # Adicionamos uma descrição para o router saber o que faz
        self.model_name = model_name
        self.system_prompt = system_prompt

    def handle_request(self, user_prompt: str) -> str:
        """Chama o seu modelo LLM específico com o prompt de sistema especializado."""
        print(f"\n>>> Invocando Expert: '{self.name}' (Modelo: {self.model_name})")
        start_time = time.time()
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ]
            )
            duration = time.time() - start_time
            print(f"<<< Resposta do Expert recebida em {duration:.2f}s")
            return response['message']['content']
        except Exception as e:
            print(f"!!! Erro ao chamar o modelo {self.model_name}: {e}")
            return f"Erro: Não foi possível obter uma resposta do expert '{self.name}'."

# ----- Router de IA (LLM-based) -----
# A nova e mais inteligente componente. Usa um LLM para a decisão.
class LLMRouter:
    def __init__(self, experts: list[Expert], router_model: str = "qwen2:0.5b"):
        self.experts = experts
        self.router_model = router_model
        # Construir o prompt do sistema para o router de uma vez para eficiência
        self.router_system_prompt = self._build_router_prompt()

    def _build_router_prompt(self) -> str:
        """
        Cria o prompt de sistema para o LLM do router.
        Este prompt instrui o modelo sobre como se comportar.
        """
        expert_list_str = "\n".join(
            [f"- nome: {e.name}, descrição: {e.description}" for e in self.experts]
        )
        
        prompt = f"""
Você é um router de tarefas inteligente e eficiente. Sua única função é analisar um pedido do utilizador e decidir qual dos seguintes experts é o mais qualificado para respondê-lo.

Lista de Experts Disponíveis:
{expert_list_str}

Responda APENAS com o nome do expert escolhido, em formato JSON. Por exemplo:
{{"expert_name": "nome_do_expert"}}
"""
        return prompt.strip()

    def select_expert(self, user_prompt: str) -> Expert:
        """
        Usa um LLM pequeno para selecionar o expert mais adequado.
        """
        print(f"\n--- Invocando Router de IA (Modelo: {self.router_model}) ---")
        print(f"Tarefa do Router: Escolher um expert para o prompt -> '{user_prompt}'")
        
        try:
            response = ollama.chat(
                model=self.router_model,
                messages=[
                    {'role': 'system', 'content': self.router_system_prompt},
                    {'role': 'user', 'content': f"Pedido do Utilizador: '{user_prompt}'"}
                ],
                options={'temperature': 0.0}, # Queremos uma resposta determinística
                format="json" # Pedimos explicitamente uma saída em JSON
            )
            
            chosen_expert_name = json.loads(response['message']['content'])['expert_name']
            print(f"--- Router de IA decidiu pelo Expert: '{chosen_expert_name}' ---")

            # Encontra o objeto Expert correspondente ao nome
            for expert in self.experts:
                if expert.name == chosen_expert_name:
                    return expert
            
            # Fallback se o LLM alucinar um nome de expert
            print(f"!!! Aviso: O router retornou um nome de expert desconhecido ('{chosen_expert_name}'). Usando fallback.")
            return next(e for e in self.experts if e.name == "general_fallback")

        except Exception as e:
            print(f"!!! Erro no Router de IA: {e}. Usando fallback.")
            return next(e for e in self.experts if e.name == "general_fallback")


# ----- Sistema MoE (quase sem alterações) -----
class MoESystem:
    def __init__(self, experts: list[Expert], router_model: str):
        self.router = LLMRouter(experts, router_model)

    def process_query(self, user_query: str) -> str:
        selected_expert = self.router.select_expert(user_query)
        response = selected_expert.handle_request(user_query)
        return response


# ----- Configuração e Execução -----
if __name__ == "__main__":
    ROUTER_MODEL = "qwen2:0.5b"
    EXPERT_MODEL = "qwen2:1.5b"
    
    # Definição dos nossos experts com as suas "personalidades" e descrições
    list_of_experts = [
        Expert(
            name="compliance_expert",
            description="Lida com questões de conformidade, leis, regulamentos e standards como GDPR, ISO 27001, NIS 2, DORA.",
            model_name=EXPERT_MODEL,
            system_prompt="Você é um especialista em conformidade de cibersegurança. Responda de forma clara e focada em regulamentos, citando normas quando possível."
        ),
        Expert(
            name="threat_analyst",
            description="Analisa ameaças, vulnerabilidades, vetores de ataque e táticas de adversários. Usa frameworks como ATT&CK, CAPEC, CWE.",
            model_name=EXPERT_MODEL,
            system_prompt="Você é um analista de Threat Intelligence. Pense como um atacante. Detalhe como uma ameaça pode ser explorada e qual o seu impacto técnico."
        ),
        Expert(
            name="controls_engineer",
            description="Sugere controlos técnicos e de processo, mitigações, e ajuda a escrever requisitos funcionais de segurança. Usa frameworks como D3FEND e NIST.",
            model_name=EXPERT_MODEL,
            system_prompt="Você é um engenheiro de segurança de software focado em soluções práticas. Sugira controlos de segurança específicos, acionáveis e realistas para mitigar as ameaças."
        ),
        Expert(
            name="general_fallback",
            description="Um assistente geral que lida com perguntas sobre cibersegurança que não se encaixam nas outras categorias.",
            model_name=EXPERT_MODEL,
            system_prompt="Você é um assistente de IA geral com um vasto conhecimento em cibersegurança. Responda de forma abrangente à pergunta do utilizador."
        )
    ]

    # Criação do nosso sistema MoE
    moe_system = MoESystem(experts=list_of_experts, router_model=ROUTER_MODEL)
    
    # --- Exemplos de Uso ---
    print("\n\n================== Exemplo 1: Conformidade ==================")
    query1 = "Quais são as implicações da DORA para uma fintech que desenvolve o seu próprio software?"
    response1 = moe_system.process_query(query1)
    print(f"\n✅ Resposta Final do Sistema:\n{response1}\n")
    
    print("\n\n================== Exemplo 2: Ameaça Específica ==================")
    query2 = "Explica-me como funciona um ataque de 'credential stuffing' e porque é perigoso."
    response2 = moe_system.process_query(query2)
    print(f"\n✅ Resposta Final do Sistema:\n{response2}\n")

    print("\n\n================== Exemplo 3: Pedido de Mitigação ==================")
    query3 = "Como posso proteger a minha aplicação web contra ataques de cross-site scripting (XSS)?"
    response3 = moe_system.process_query(query3)
    print(f"\n✅ Resposta Final do Sistema:\n{response3}\n")
    
    print("\n\n================== Exemplo 4: Pergunta Ampla (deve usar o fallback) ==================")
    query4 = "Qual é o papel da criptografia pós-quântica no futuro da cibersegurança?"
    response4 = moe_system.process_query(query4)
    print(f"\n✅ Resposta Final do Sistema:\n{response4}\n")
