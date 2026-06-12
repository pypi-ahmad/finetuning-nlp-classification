"""
triage_model.py
─────────────────────────────────────────────────────────────────────────────
Prompts, the hierarchical label scorer, and the multi-task SFT record builder
for the two-stage triage model.

ONE QLoRA adapter does both stages (multi-task instruction tuning):
    Stage 1 (L1): given the complaint + the product rubric → predict the product.
    Stage 2 (L2): given the complaint + the *issues under a product* → predict
                  the issue, conditioned on the L1 chosen in stage 1.

We never let the model free-generate. For each stage we **score the candidate
labels** as completions and softmax their length-normalised log-likelihoods →
calibrated probabilities (→ top-k + confidence). This guarantees valid labels
and works identically for the base and fine-tuned models. Scoring uses a
memory-frugal `logit − logsumexp` so the 152k-token vocab fits 8 GB.
"""

from __future__ import annotations

import numpy as np
import torch

from triage_config import CONFIG, TriageConfig

# Short product glosses for the L1 rubric (keys match download_cfpb.CANONICAL_PRODUCTS)
L1_GLOSS: dict[str, str] = {
    "Credit reporting & repair": "credit reports/scores, disputes with bureaus, repair services",
    "Debt collection": "collectors pursuing a debt: validation, contact, harassment",
    "Mortgage": "home loans: servicing, escrow, payments, modification, foreclosure",
    "Credit card / prepaid": "credit & prepaid cards: charges, billing, rewards, fees",
    "Bank account / savings": "checking/savings: deposits, overdrafts, fees, account access",
    "Money transfer & virtual currency": "transfers, wires, P2P, crypto, money services",
    "Student loan": "federal/private student loans: servicing, repayment",
    "Vehicle / consumer loan": "auto loans/leases and other consumer installment loans",
    "Payday / title / personal loan": "short-term, title, personal or advance loans",
}


def _bullets(items: list[str], gloss: dict[str, str] | None = None) -> str:
    out = []
    for it in items:
        if gloss and it in gloss:
            out.append(f"- {it}: {gloss[it]}")
        else:
            out.append(f"- {it}")
    return "\n".join(out)


def l1_system(l1_labels: list[str]) -> str:
    return (
        "You are the FIRST stage of a financial-complaint triage system. "
        "Read the complaint and choose the single best PRODUCT category. "
        "Reply with the category name only.\n\nProduct categories:\n"
        + _bullets(l1_labels, L1_GLOSS)
    )


def l2_system(l1: str, l2_candidates: list[str]) -> str:
    return (
        f"You are the SECOND stage of a financial-complaint triage system. "
        f"The complaint has been routed to the product \"{l1}\". "
        f"Choose the single best ISSUE within this product. "
        f"Reply with the issue name only.\n\nIssues for {l1}:\n"
        + _bullets(l2_candidates)
    )


def l1_messages(text: str, l1_labels: list[str]) -> list[dict]:
    return [{"role": "system", "content": l1_system(l1_labels)},
            {"role": "user", "content": f'Complaint: "{text.strip()}"'}]


def l2_messages(text: str, l1: str, l2_candidates: list[str]) -> list[dict]:
    return [{"role": "system", "content": l2_system(l1, l2_candidates)},
            {"role": "user", "content": f'Complaint: "{text.strip()}"'}]


# ─────────────────────────────────────────────────────────────────────────────
# Multi-task SFT records (string prompt/completion → completion_only_loss masks
# the prompt; verified to truncate correctly at CONFIG.max_length)
# ─────────────────────────────────────────────────────────────────────────────
def make_sft_records(train_df, hierarchy, tokenizer, cfg: TriageConfig = CONFIG):
    """Build a mixed list of L1 and L2 instruction examples from the train split."""
    from datasets import Dataset

    l1_labels = sorted(hierarchy.keys())
    records = []
    for row in train_df.itertuples(index=False):
        # Stage-1 example for every row (CFPB + Banking77)
        p1 = tokenizer.apply_chat_template(
            l1_messages(row.text, l1_labels), tokenize=False, add_generation_prompt=True)
        records.append({"prompt": p1, "completion": row.l1})
        # Stage-2 example only when we have a valid L2 under this L1
        if row.l2 is not None and row.l2 in hierarchy.get(row.l1, []):
            cands = hierarchy[row.l1]
            p2 = tokenizer.apply_chat_template(
                l2_messages(row.text, row.l1, cands), tokenize=False, add_generation_prompt=True)
            records.append({"prompt": p2, "completion": row.l2})
    return Dataset.from_list(records)


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical scorer
# ─────────────────────────────────────────────────────────────────────────────
class HierScorer:
    """Score candidate labels as completions; returns per-text probability dicts."""

    def __init__(self, model, tokenizer, cfg: TriageConfig = CONFIG):
        self.model = model
        self.tok = tokenizer
        self.cfg = cfg

    def _prompt_ids(self, messages) -> list[int]:
        out = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True)
        return list(out["input_ids"])

    @torch.no_grad()
    def _avg_logprobs(self, prompt_ids_list, cand_ids_list) -> list[float]:
        """Length-normalised log-likelihood of each (prompt, candidate) pair."""
        device = next(self.model.parameters()).device
        pad_id = self.tok.pad_token_id or self.tok.eos_token_id
        scores = [0.0] * len(prompt_ids_list)
        bs = self.cfg.eval_batch_size
        for s in range(0, len(prompt_ids_list), bs):
            P = prompt_ids_list[s:s + bs]
            C = cand_ids_list[s:s + bs]
            full = [p + c for p, c in zip(P, C)]
            maxlen = max(len(x) for x in full)
            ids, attn, cmask = [], [], []
            for p, c, seq in zip(P, C, full):
                pad = maxlen - len(seq)
                ids.append(seq + [pad_id] * pad)
                attn.append([1] * len(seq) + [0] * pad)
                cmask.append([0] * len(p) + [1] * len(c) + [0] * pad)
            ids = torch.tensor(ids, device=device)
            attn = torch.tensor(attn, device=device)
            cmask = torch.tensor(cmask, device=device, dtype=torch.float32)[:, 1:]
            logits = self.model(input_ids=ids, attention_mask=attn).logits[:, :-1, :]
            tgt = ids[:, 1:]
            tok_lp = (logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                      - torch.logsumexp(logits, dim=-1)).float()
            del logits
            sum_lp = (tok_lp * cmask).sum(1)
            n = cmask.sum(1).clamp(min=1)
            for j, v in enumerate((sum_lp / n).cpu().numpy()):
                scores[s + j] = float(v)
        return scores

    def raw_scores(self, messages_per_text, candidates_per_text) -> list[dict]:
        """Per-text {label: raw length-normalised log-likelihood} (pre-softmax).

        messages_per_text   : list of chat-message lists (caller builds the prompt)
        candidates_per_text : list of candidate-label lists to score as completions
        """
        prompt_ids_list, cand_ids_list, owner = [], [], []
        cache_cand_ids: dict[str, list[int]] = {}
        for ti, (msgs, cands) in enumerate(zip(messages_per_text, candidates_per_text)):
            p_ids = self._prompt_ids(msgs)
            for c in cands:
                if c not in cache_cand_ids:
                    cache_cand_ids[c] = self.tok(c, add_special_tokens=False)["input_ids"]
                prompt_ids_list.append(p_ids)
                cand_ids_list.append(cache_cand_ids[c])
                owner.append((ti, c))
        avg = self._avg_logprobs(prompt_ids_list, cand_ids_list)
        per_text: list[dict] = [dict() for _ in messages_per_text]
        for (ti, c), v in zip(owner, avg):
            per_text[ti][c] = v
        return per_text

    @staticmethod
    def softmax_dict(d: dict, temperature: float = 1.0) -> dict:
        labels = list(d.keys())
        arr = np.array([d[l] for l in labels]) / temperature
        arr = arr - arr.max()
        p = np.exp(arr); p /= p.sum()
        return {l: float(pi) for l, pi in zip(labels, p)}

    def score(self, messages_per_text, candidates_per_text,
              temperature: float = 1.0) -> list[dict]:
        """Per-text {label: probability} (softmax over that text's candidate set)."""
        return [self.softmax_dict(d, temperature)
                for d in self.raw_scores(messages_per_text, candidates_per_text)]

    # ── convenience wrappers for the two stages ──────────────────────────────
    def score_l1(self, texts, l1_labels, temperature: float = 1.0) -> list[dict]:
        msgs = [l1_messages(t, l1_labels) for t in texts]
        return self.score(msgs, [l1_labels] * len(texts), temperature)

    def score_l2(self, texts, l1s, hierarchy, temperature: float = 1.0) -> list[dict]:
        msgs = [l2_messages(t, l1, hierarchy[l1]) for t, l1 in zip(texts, l1s)]
        cands = [hierarchy[l1] for l1 in l1s]
        return self.score(msgs, cands, temperature)

    def l1_score_matrix(self, texts, l1_labels):
        """Return (N, K) raw-score matrix aligned to l1_labels (fixed candidate set)."""
        msgs = [l1_messages(t, l1_labels) for t in texts]
        raw = self.raw_scores(msgs, [l1_labels] * len(texts))
        return np.array([[d[l] for l in l1_labels] for d in raw])


def hier_evaluate(scorer: "HierScorer", df, hierarchy, l1_temperature: float = 1.0,
                  l1_score_matrix=None):
    """
    Two-stage evaluation. Returns a metrics dict plus arrays for error analysis.

    Metrics:
      l1_macro_f1, l1_accuracy, l1_top3_accuracy   (all rows)
      l2_macro_f1, l2_accuracy                      (CFPB rows, L2 under GOLD L1)
      hier_exact_match                              (CFPB rows: L1 right AND L2 right)

    Note: hierarchical exact-match only credits L2 when L1 is correct, and when
    L1 is correct the gold- and predicted-L1 candidate sets are identical — so a
    single L2 pass under the gold L1 yields both the isolated L2 score and EM.
    """
    from sklearn.metrics import f1_score, accuracy_score

    l1_labels = sorted(hierarchy.keys())
    texts = list(df["text"])

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    S = l1_score_matrix if l1_score_matrix is not None else scorer.l1_score_matrix(texts, l1_labels)
    s = S / l1_temperature
    s = s - s.max(1, keepdims=True)
    l1_probs = np.exp(s); l1_probs /= l1_probs.sum(1, keepdims=True)
    pred_l1_idx = l1_probs.argmax(1)
    pred_l1 = [l1_labels[i] for i in pred_l1_idx]
    gold_l1 = list(df["l1"])
    gold_l1_idx = np.array([l1_labels.index(g) for g in gold_l1])

    top3 = np.argsort(-l1_probs, axis=1)[:, :3]
    l1_top3 = float(np.mean([g in row for g, row in zip(gold_l1_idx, top3)]))

    out = {
        "l1_macro_f1": float(f1_score(gold_l1, pred_l1, average="macro")),
        "l1_accuracy": float(accuracy_score(gold_l1, pred_l1)),
        "l1_top3_accuracy": l1_top3,
    }

    # ── Stage 2 (CFPB rows with a valid issue under their gold L1) ────────────
    mask = [(src == "cfpb" and l2 is not None and l2 in hierarchy.get(l1, []))
            for src, l2, l1 in zip(df["source"], df["l2"], df["l1"])]
    sub = df[np.array(mask)].reset_index(drop=True)
    pred_l1_for_sub = np.array(pred_l1)[np.array(mask)]
    if len(sub):
        l2_scored = scorer.score_l2(list(sub["text"]), list(sub["l1"]), hierarchy)
        pred_l2 = [max(p, key=p.get) for p in l2_scored]
        gold_l2 = list(sub["l2"])
        out["l2_macro_f1"] = float(f1_score(gold_l2, pred_l2, average="macro"))
        out["l2_accuracy"] = float(accuracy_score(gold_l2, pred_l2))
        l1_correct = (pred_l1_for_sub == np.array(list(sub["l1"])))
        l2_correct = np.array([p == g for p, g in zip(pred_l2, gold_l2)])
        out["hier_exact_match"] = float(np.mean(l1_correct & l2_correct))
    else:
        out.update(l2_macro_f1=float("nan"), l2_accuracy=float("nan"),
                   hier_exact_match=float("nan"))

    extras = {"l1_probs": l1_probs, "pred_l1": pred_l1, "gold_l1": gold_l1,
              "l1_labels": l1_labels, "l1_score_matrix": S}
    return out, extras


def load_base_model(cfg: TriageConfig = CONFIG, adapter_dir=None):
    """Load Qwen in 4-bit; optionally attach a saved LoRA adapter."""
    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, quantization_config=bnb, device_map="cuda")
    if adapter_dir and Path(adapter_dir).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tok
