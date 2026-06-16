import requests
import json
import re
from typing import Optional, Dict, Any, Tuple


class OllamaModel:
    def __init__(
        self,
        model_name: str = "gemma2:9b",
        host: str = "http://127.0.0.1:11434/api/generate",
        num_thread: int = 10,
        num_ctx: int = 4096,
        temperature: float = 0.2,
        top_p: float = 0.9,
        seed: int | None = None,
    ) -> None:
        
        self.llm_model = model_name
        self.ollama_url = host
        self.num_thread = num_thread
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed

    def _get_complete_query(self, system_prompt: str, user_prompt: str) -> str:
        return f"""
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        {system_prompt}
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {user_prompt}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """.strip()

    def _parse_llm_answer(self, response: requests.Response) -> str:
        try:
            data = response.json()
            answer = data.get("response", "")
        except Exception:
            answer = ""
            for line in response.text.split("\n"):
                try:
                    answer += json.loads(line).get("response", "")
                except Exception:
                    continue

        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        return answer

    def generation(
        self,
        system_prompt: str,
        user_prompt: str,
        token_budget: Optional[int] = 1024,
    ) -> Tuple[str, Dict[str, Any]]:
        
        complete_query = self._get_complete_query(system_prompt, user_prompt)

        options = {
            "num_thread": self.num_thread,
            "num_ctx": self.num_ctx,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
        }

        if self.seed is not None:
            options["seed"] = self.seed

        if token_budget is not None:
            options["num_predict"] = token_budget

        payload = {
            "model": self.llm_model,
            "prompt": complete_query,
            "stream": False,
            "options": options,
        }

        response = requests.post(self.ollama_url, json=payload, timeout=300)
        response.raise_for_status()

        data = response.json()
        answer = self._parse_llm_answer(response)

        total_duration_s = data.get("total_duration", 0) / 1e9
        load_duration_s = data.get("load_duration", 0) / 1e9
        prompt_eval_duration_s = data.get("prompt_eval_duration", 0) / 1e9
        eval_duration_s = data.get("eval_duration", 0) / 1e9

        eval_count = data.get("eval_count", 0) or 0
        prompt_eval_count = data.get("prompt_eval_count", 0) or 0

        tokens_per_second = (
            eval_count / eval_duration_s if eval_duration_s and eval_duration_s > 0 else 0.0
        )
        request_metrics = {
            "model": self.llm_model,
            "seed": self.seed,

            "prompt_tokens": prompt_eval_count,
            "response_tokens": eval_count,
            "total_tokens": prompt_eval_count + eval_count,

            "total_duration_s": total_duration_s,
            "load_duration_s": load_duration_s,
            "prompt_eval_duration_s": prompt_eval_duration_s,
            "eval_duration_s": eval_duration_s,
            "tokens_per_second": tokens_per_second,

            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
            "total_tokens_observed": prompt_eval_count + eval_count,
        }

        return answer, request_metrics