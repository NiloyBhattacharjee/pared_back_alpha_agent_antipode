from __future__ import annotations

"""
LangChain-powered news sentiment agent (optional dependencies).

Behavior
- Filters headlines for each ticker within the trailing `lookback_days` and up to `as_of`.
- Attempts retrieval via a local embedding vector store (FAISS + MiniLM). If
  LangChain/embeddings are unavailable, falls back to a simple recency-k
  selection (RAG-lite).
- Scoring hierarchy per ticker:
  1) If OPENAI_API_KEY is present and langchain_openai is installed, ask an LLM
     to return a scalar in [-1, 1] for net sentiment from the retrieved
     snippets.
  2) Otherwise, fall back to a transparent lexicon score identical in spirit
     to NewsSentimentAgent: (pos - neg) / (pos + neg) with a tiny lexicon in
     src.data (optionally you can extend it).

This module is entirely optional. If imports fail, callers can catch and fall
back to the existing NewsSentimentAgent.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from .data import POS_WORDS, NEG_WORDS


def _safe_import_langchain():
    try:
        from langchain_community.vectorstores import FAISS  # type: ignore
        from langchain_community.embeddings import (
            HuggingFaceEmbeddings,  # type: ignore
        )
        return FAISS, HuggingFaceEmbeddings
    except Exception:
        return None, None


def _maybe_llm_score(snippets: List[str]) -> Optional[float]:
    import os

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
        from langchain.prompts import PromptTemplate  # type: ignore

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=key)
        prompt = PromptTemplate.from_template(
            "You are an equity analyst. Given these recent company headlines:\n"
            "{snips}\n"
            "Return ONE number only in [-1,1] for net investor sentiment."
        )
        text = "\n".join(f"- {s}" for s in snippets[:12])
        resp = llm.invoke(prompt.format(snips=text))
        out = str(getattr(resp, "content", resp)).strip()
        try:
            return float(out)
        except Exception:
            return None
    except Exception:
        return None


def _lex_score(text: str) -> float:
    t = text.lower()
    pos = sum(1 for w in POS_WORDS if w in t)
    neg = sum(1 for w in NEG_WORDS if w in t)
    if pos == 0 and neg == 0:
        return 0.0
    return float(pos - neg) / float(pos + neg)


@dataclass
class LangChainNewsAgent:
    lookback_days: int = 60
    k: int = 20

    def _build_vs(self, news: pd.DataFrame):
        FAISS, HFEmb = _safe_import_langchain()
        if FAISS is None or HFEmb is None:
            return None
        try:
            embeddings = HFEmb(model_name="sentence-transformers/all-MiniLM-L6-v2")
            texts = news["headline"].astype(str).tolist()
            metas = news[["ticker", "date"]].to_dict(orient="records")
            vs = FAISS.from_texts(texts, embeddings, metadatas=metas)
            return vs
        except Exception:
            return None

    def score(self, news: pd.DataFrame, as_of: datetime) -> pd.Series:
        if news.empty:
            return pd.Series(dtype=float)
        df = news.copy()
        df = df[(df["date"] <= pd.to_datetime(as_of.date())) & (
            df["date"] >= pd.to_datetime(as_of.date()) - pd.Timedelta(days=self.lookback_days)
        )]
        if df.empty:
            return pd.Series(dtype=float)

        # Build vector store if possible; otherwise fall back to recency selection
        vs = self._build_vs(df)
        out = {}
        for t in sorted(df["ticker"].unique()):
            sub = df[df["ticker"] == t].copy()
            if sub.empty:
                out[t] = np.nan
                continue

            snippets: List[str]
            if vs is not None:
                try:
                    retriever = vs.as_retriever(search_kwargs={"k": self.k, "filter": {"ticker": t}})  # type: ignore
                    docs = retriever.invoke(
                        f"{t} company updates demand supply guidance earnings regulation risk partnership"
                    )
                    snippets = [getattr(d, "page_content", str(d)) for d in docs]
                except Exception:
                    snippets = []
            else:
                # RAG-lite: most recent k headlines
                sub = sub.sort_values("date").tail(self.k)
                snippets = sub["headline"].astype(str).tolist()

            if not snippets:
                out[t] = np.nan
                continue

            # LLM pathway (optional)
            llm_val = _maybe_llm_score(snippets)
            if llm_val is not None:
                out[t] = float(np.clip(llm_val, -1.0, 1.0))
                continue

            # Transparent lexicon fallback (weighted by simple recency if available)
            if vs is None:
                # weight by recency decay; compute vectorized day differences robustly
                sub_sorted = sub.sort_values("date")
                dates_norm = pd.to_datetime(sub_sorted["date"]).dt.normalize()
                ages = (pd.Timestamp(as_of.date()) - dates_norm).dt.days.astype(int)
                w = (0.95 ** ages.to_numpy(dtype=float))
                w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
                sents = np.array([_lex_score(s) for s in snippets], dtype=float)
                out[t] = float(np.dot(sents, w))
            else:
                # unweighted mean of retrieved snippets
                sents = np.array([_lex_score(s) for s in snippets], dtype=float)
                out[t] = float(np.mean(sents))

        return pd.Series(out)

    def rate(self, news: pd.DataFrame, as_of: datetime):
        s = self.score(news, as_of)
        ratings = s.apply(lambda x: "BUY" if x >= 0.1 else ("SELL" if x <= -0.1 else "HOLD"))
        return s, ratings
