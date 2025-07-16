import ollama
import json
import re
import time
import datetime

# ==============================================================================
#  Global Configuration for Logging
# ==============================================================================
LOG_FILE_PATH = "moe_system_log.txt"

def write_log(log_content: str):
    """Appends a string to the log file with a timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(f"--- Log Entry: {timestamp} ---\n")
        f.write(log_content)
        f.write("\n" + "="*80 + "\n\n")

# ==============================================================================
#  Expert and Router Classes (unchanged from the previous robust version)
# ==============================================================================
class Expert:
    # ... (code is identical to the previous English version)
    def __init__(self, name: str, description: str, model_name: str, system_prompt: str):
        self.name = name; self.description = description; self.model_name = model_name; self.system_prompt = system_prompt
    def handle_request(self, user_prompt: str) -> str:
        print(f"\n>>> Invoking Expert: '{self.name}' (Model: {self.model_name})")
        start_time = time.time()
        try:
            response = ollama.chat(model=self.model_name, messages=[{'role': 'system', 'content': self.system_prompt},{'role': 'user', 'content': user_prompt}])
            duration = time.time() - start_time
            print(f"<<< Expert response received in {duration:.2f}s")
            return response['message']['content']
        except Exception as e:
            print(f"!!! Error calling model {self.model_name}: {e}"); return f"Error: Could not get a response from expert '{self.name}'."

class LLMRouter:
    # ... (code is identical to the previous English version)
    def __init__(self, experts: list[Expert], router_model: str = "qwen3:4b"):
        self.experts = experts; self.router_model = router_model; self.valid_expert_names = [e.name for e in experts]; self.router_system_prompt = self._build_router_prompt()
    def _build_router_prompt(self) -> str:
        expert_list_str = "\n".join([f"- {e.name}: {e.description}" for e in self.experts])
        prompt = f"""
You are a highly-accurate task routing AI. Your sole purpose is to analyze a user's request and determine which of the following experts is best qualified to handle it.
Available Experts:
{expert_list_str}
**Critical Instructions:**
1. Analyze the user's request.
2. Choose the EXACT NAME of ONE expert from the list above.
3. Your response MUST be a JSON object containing ONLY the key "expert_name".
**Example of a perfect response:**
{{"expert_name": "compliance_expert"}}
DO NOT include any explanations, comments, or any other text outside of the JSON object. Your final output must be only the JSON.
"""; return prompt.strip()
    def _parse_and_validate_response(self, response_content: str) -> str | None:
        try:
            data = json.loads(response_content)
            if 'expert_name' in data and data['expert_name'] in self.valid_expert_names: return data['expert_name']
        except json.JSONDecodeError:
            print("Router > Response was not valid JSON. Attempting regex extraction...")
            for name in self.valid_expert_names:
                if re.search(f"['\"]{name}['\"]", response_content): return name
        return None
    def select_expert(self, user_prompt: str) -> tuple[Expert, dict]:
        print(f"\n--- Invoking AI Router (Model: {self.router_model}) ---"); print(f"Router Task: Select expert for prompt -> '{user_prompt}'")
        log_data = {"router_raw_response": "N/A", "validated_choice": "N/A"}
        try:
            response = ollama.chat(model=self.router_model, messages=[{'role': 'system', 'content': self.router_system_prompt},{'role': 'user', 'content': f"User Request: \"{user_prompt}\""}], options={'temperature': 0.0}, format="json")
            response_content = response['message']['content']; log_data["router_raw_response"] = response_content; print(f"--- AI Router Raw Response: {response_content} ---")
            chosen_expert_name = self._parse_and_validate_response(response_content)
            if chosen_expert_name:
                log_data["validated_choice"] = chosen_expert_name; print(f"--- AI Router Validated Choice: '{chosen_expert_name}' ---")
                return next(e for e in self.experts if e.name == chosen_expert_name), log_data
            print(f"!!! Warning: Router returned an invalid or unparsable response. Using fallback.")
            log_data["validated_choice"] = "general_fallback (Forced)"; return next(e for e in self.experts if e.name == "general_fallback"), log_data
        except Exception as e:
            print(f"!!! Critical AI Router Error: {e}. Using fallback.")
            log_data["validated_choice"] = "general_fallback (Critical Error)"; return next(e for e in self.experts if e.name == "general_fallback"), log_data


class MoESystem:
    def __init__(self, experts: list[Expert], router_model: str):
        self.router = LLMRouter(experts, router_model)

    def process_query(self, user_query: str) -> tuple[str, dict]:
        """Processes a query and returns the final response and the routing log."""
        selected_expert, routing_log = self.router.select_expert(user_query)
        response = selected_expert.handle_request(user_query)
        return response, routing_log

# ==============================================================================
#  Main Execution Block (with evaluation and logging)
# ==============================================================================
if __name__ == "__main__":
    # Clear log file at the start of the run
    with open(LOG_FILE_PATH, "w") as f:
        f.write(f"MoE System Log Initialized at {datetime.datetime.now()}\n\n")

    # --- System Setup (unchanged) ---
    ROUTER_MODEL = "qwen3:4b"
    EXPERT_MODEL = "qwen3:4b"
    list_of_experts = [ Expert(name="compliance_expert", description="Handles compliance, laws, standards like GDPR, ISO 27001.", model_name=EXPERT_MODEL, system_prompt="You are a cybersecurity compliance specialist."), Expert(name="threat_analyst", description="Analyzes threats, attack vectors, adversary tactics using ATT&CK, CAPEC.", model_name=EXPERT_MODEL, system_prompt="You are a Threat Intelligence Analyst."), Expert(name="controls_engineer", description="Suggests technical controls, mitigations, and functional requirements.", model_name=EXPERT_MODEL, system_prompt="You are a Software Security Engineer."), Expert(name="general_fallback", description="A general assistant for broader cybersecurity questions.", model_name=EXPERT_MODEL, system_prompt="You are a general AI assistant for cybersecurity.") ]
    moe_system = MoESystem(experts=list_of_experts, router_model=ROUTER_MODEL)
    
    # --- Evaluation Dataset: Queries and their "Gold-Standard" answers ---
    evaluation_set = [
        {
            "id": 1,
            "query": "What are the implications of the DORA act for a fintech company developing its own software?",
            "expected_expert": "compliance_expert",
            "gold_standard_answer": "DORA requires a comprehensive ICT risk framework, rigorous resilience testing for in-house software (Art. 24), management of third-party dependencies (like open-source libraries, Art. 28), and strict incident reporting mechanisms (Art. 17)."
        },
        {
            "id": 2,
            "query": "Explain to me how a 'credential stuffing' attack works and why it is dangerous.",
            "expected_expert": "threat_analyst",
            "gold_standard_answer": "A credential stuffing attack (T1110.003) uses large, automated lists of username/password pairs stolen from other data breaches. The danger is a scalable Account Takeover (ATO) where thousands of accounts reusing passwords can be compromised quickly and often stealthily by distributing attacks across many IP addresses."
        },
        {
            "id": 3,
            "query": "How can I protect my web application from cross-site scripting (XSS) attacks?",
            "expected_expert": "controls_engineer",
            "gold_standard_answer": "The primary defenses against XSS are: 1) Contextual Output Encoding of all user-supplied data before rendering. 2) Implementing a strict Content Security Policy (CSP) header. 3) Setting the HttpOnly flag on session cookies to prevent script access."
        },
        {
            "id": 4,
            "query": "What is the role of post-quantum cryptography in the future of cybersecurity?",
            "expected_expert": "general_fallback",
            "gold_standard_answer": "Post-quantum cryptography (PQC) aims to provide new cryptographic algorithms secure against attacks from both classical and future quantum computers, which threaten to break current public-key crypto like RSA. Its role is to future-proof digital communications against 'harvest now, decrypt later' attacks."
        }
    ]

    # --- Main Loop: Process each query, log, and print comparison ---
    for item in evaluation_set:
        print(f"\n\n{'='*25} Processing Evaluation Item #{item['id']} {'='*25}")
        
        # Process the query
        generated_response, routing_log = moe_system.process_query(item["query"])
        
        # --- Prepare the full log content ---
        log_output = f"""
Query ID: {item['id']}
User Query: "{item['query']}"

[ROUTING]
Expected Expert: "{item['expected_expert']}"
Router Raw Response: "{routing_log['router_raw_response']}"
Router Validated Choice: "{routing_log['validated_choice']}"
Routing Correct: {item['expected_expert'] == routing_log['validated_choice']}

[RESPONSE]
Generated Response:
{generated_response}

Gold-Standard (Correct) Answer:
{item['gold_standard_answer']}
"""
        # Print to terminal
        print("\n" + "-"*20 + " EVALUATION & LOG " + "-"*20)
        print(log_output)
        
        # Write to log file
        write_log(log_output)
        
    print(f"\n\n✅ Evaluation complete. All results saved to '{LOG_FILE_PATH}'.")