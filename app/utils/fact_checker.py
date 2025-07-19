import json
import re
from typing import List, Dict, Optional, Any

import numpy as np
import pandas as pd
from openai import OpenAI
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from app.prompts.claim_extraction import FULL_PROMPT
from app.fact_verification import create_claim, VerificationResult

class FactChecker:
    """Fact-checking engine that extracts claims using GPT and validates them."""

    _EXTRACTION_PROMPT = FULL_PROMPT

    def __init__(self, anndata_model, openai_client: Optional[OpenAI] = None):
        self.anndata_model = anndata_model
        self.client: Optional[OpenAI] = openai_client

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def set_client(self, client: OpenAI):
        """Attach an OpenAI client after initialization."""
        self.client = client

    def validate_response(self, response_text: str) -> List[Dict]:
        """Extract claims via GPT, convert to claim objects, verify against dataset.

        Returns list of dicts: {claim: <text>, result: 'TRUE'|'FALSE'|'NOT_VERIFIABLE', reason: <str|None>}"""
        raw_claims = self.extract_claims(response_text)
        results: List[Dict] = []

        for raw in raw_claims:
            text_repr = raw.get("text") or raw.get("original_text") or json.dumps(raw)
            try:
                claim_obj = create_claim(raw)
            except Exception as e:
                results.append({
                    "claim": text_repr,
                    "result": "NOT_VERIFIABLE",
                    "reason": f"parse_error: {str(e)}",
                })
                continue

            try:
                outcome = claim_obj.verify(self.anndata_model.adata if self.anndata_model else None)
            except Exception as e:
                results.append({
                    "claim": text_repr,
                    "result": "NOT_VERIFIABLE",
                    "reason": f"verification_exception: {str(e)}",
                })
                continue

            if outcome.is_true:
                results.append({"claim": text_repr, "result": "TRUE", "reason": outcome.reason})
            elif outcome.veryfiable:
                results.append({"claim": text_repr, "result": "FALSE", "reason": f'{round(outcome.true_ratio, 4)*100}% of cells satisfy condition'})
            else:
                results.append({"claim": text_repr, "result": "NOT_VERIFIABLE", "reason": outcome.reason})
        return results

    # ------------------------------------------------------------------
    # Internal logic
    # ------------------------------------------------------------------
    def extract_claims(self, chat_response: str) -> List[Dict]:
        """Use GPT to turn chat response into structured claim list."""
        if self.client is None:
            return []  # cannot parse without OpenAI

        metadata_summary = self._build_metadata_summary()
        messages = []
        # Compose messages
        if metadata_summary:
            messages.append({
                "role": "system",
                "content": "Dataset metadata summary (for reference):\n" + metadata_summary,
            })


        messages.append({"role": "system", "content": self._EXTRACTION_PROMPT})

        

        messages.append({"role": "user", "content": f"```{chat_response}```"})
        try:
            completion = self.client.chat.completions.create(
                # model="gpt-4o-mini",  # lightweight model enough for extraction
                model='gpt-4.1-2025-04-14',
                messages=messages,
                temperature=0,
            )
            content = completion.choices[0].message.content
            print(completion.choices[0].message.content)
            # ensure JSON
            json_start = content.find("[")
            json_end = content.rfind("]") + 1
            if json_start == -1 or json_end == -1:
                return []
            json_str = content[json_start:json_end]
            claims = json.loads(json_str)
            # attach original_text for reference
            for item in claims:
                item.setdefault("original_text", json.dumps(item))
            return claims
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Metadata helper
    # ------------------------------------------------------------------
    def _build_metadata_summary(self) -> str:
        """Create a textual summary of adata.obs columns for prompting."""
        if self.anndata_model is None or self.anndata_model.adata is None:
            return ""

        obs = self.anndata_model.adata.obs
        summary_lines: List[str] = []
        for col in obs.columns:
            series = obs[col]
            if is_bool_dtype(series):
                uniques = series.unique()
                line = f"{col} (boolean): {', '.join(map(str, uniques))}"
            elif is_numeric_dtype(series):
                desc = series.describe()
                line = (
                    f"{col} (numeric): min {desc['min']:.4g}, 25% {desc['25%']:.4g}, "
                    f"median {desc['50%']:.4g}, 75% {desc['75%']:.4g}, max {desc['max']:.4g}"
                )
            else:
                uniques = series.unique()
                # Limit to first 30 values to keep prompt short
                listed = ", ".join(map(str, uniques[:30]))
                if len(uniques) > 30:
                    listed += ", ..."
                line = f"{col} (categorical): {listed}"
            summary_lines.append(line)

        return "\n".join(summary_lines)

    # remove old _validate_claim implementation (deprecated) 