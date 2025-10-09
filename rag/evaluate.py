import json
from typing import Any, Dict, List, Union

import ollama


class RAGEvaluator:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def _ensure_list(
        self, data_or_path: Union[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        if isinstance(data_or_path, str):
            with open(data_or_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, list):
                raise ValueError("Input JSON must be a list of objects")
            return loaded
        return data_or_path

    def _judge_relevance(self, query: str, chunk_text: str) -> int:
        prompt = (
            "You are a strict relevance judge.\n"
            "Given a search query and one candidate text chunk, respond with exactly one character: 1 if the chunk is relevant to answering the query, otherwise 0.\n"
            "No explanation and no extra characters.\n\n"
            f"Query: {query}\n"
            "Chunk:\n"
            f"{chunk_text}\n\n"
            "Answer (0 or 1):"
        )
        try:
            result = ollama.generate(model=self.model_name, prompt=prompt)
            response = (result.get("response") or "").strip()
            return 1 if response.startswith("1") else 0
        except Exception:
            return 0

    def evaluate(
        self, data_or_path: Union[str, List[Dict[str, Any]]]
    ) -> Dict[str, Dict[str, int]]:
        items = self._ensure_list(data_or_path)
        evaluations: Dict[str, Dict[str, int]] = {}

        for obj in items:
            query = obj.get("query", "").strip()
            query_id = str(obj.get("query_num", len(evaluations) + 1))
            chunk_scores: Dict[str, int] = {}

            chunk_indices: List[int] = []
            for key in obj.keys():
                if key.startswith("chunk_") and key.endswith("_text"):
                    try:
                        idx = int(key.split("_")[1])
                        chunk_indices.append(idx)
                    except Exception:
                        continue
            chunk_indices.sort()

            for idx in chunk_indices:
                text_key = f"chunk_{idx}_text"
                id_key = f"chunk_{idx}_chunk_id"
                file_key = f"chunk_{idx}_filename"

                chunk_text = str(obj.get(text_key, ""))
                if not chunk_text.strip():
                    continue

                chunk_identifier = (
                    str(obj.get(id_key))
                    if obj.get(id_key) is not None
                    else (
                        str(obj.get(file_key))
                        if obj.get(file_key) is not None
                        else f"chunk_{idx}"
                    )
                )

                relevance = self._judge_relevance(query=query, chunk_text=chunk_text)
                chunk_scores[chunk_identifier] = relevance

            evaluations[query_id] = chunk_scores

        return evaluations
