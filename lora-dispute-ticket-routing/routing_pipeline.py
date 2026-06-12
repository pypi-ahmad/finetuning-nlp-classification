"""
routing_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Shared, importable building blocks for the dispute-routing project so the
notebook and inference.py never drift apart:

    * RoutingConfig         – one place for model id / hyper-params / paths
    * load_routing_dataset  – Banking77 → 8 routing queues, stratified splits
    * build_messages        – the chat prompt (system rubric + user complaint)
    * RouteScorer           – constrained label-likelihood scorer → calibrated
                              probabilities over the 8 routes (top-1 / top-3 /
                              confidence / calibration all come from this)
    * to_sft_dataset        – prompt/completion format for TRL SFTTrainer
    * routing_metrics       – accuracy, macro-F1, top-3 accuracy

Design choice — *why score labels instead of free-generating one?*
    A router has a fixed, known set of destinations. Instead of letting the LLM
    generate arbitrary text and hoping it matches a route, we ask the model how
    likely each of the 8 route strings is as the completion, then softmax those
    scores. This guarantees a valid label, gives a real probability per route
    (→ top-3 + calibration), and works identically for the zero-shot baseline
    and the fine-tuned adapter. We length-normalise the log-likelihood so the
    model isn't biased toward shorter route names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from routing_taxonomy import (
    ROUTES,
    ROUTE_DESCRIPTIONS,
    ROUTE2ID,
    ID2ROUTE,
    route_for_intent,
    assert_full_coverage,
)

HERE = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RoutingConfig:
    # Model — 1.5B instruct fits 8 GB comfortably in 4-bit QLoRA.
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Data sizing (kept modest so a full run finishes on a laptop GPU).
    train_size: int = 5000          # stratified sample of Banking77 train
    val_size: int = 800             # held-out from train for early signal
    test_size: int = 1200           # stratified sample of Banking77 test
    seed: int = 42

    # Tokenisation. The routing rubric (system prompt) is ~265 tokens, so a full
    # prompt+completion runs ~300–380 tokens. max_length MUST exceed that or the
    # collator's keep_start truncation cuts the completion → zero supervised
    # tokens → loss never moves. (Learned the hard way; see notebook §6.)
    max_length: int = 384

    # LoRA / QLoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # Training (8 GB-friendly)
    epochs: float = 2.0
    per_device_batch_size: int = 4  # 8 OOMs: 152k-vocab loss tensor is the limit
    grad_accum: int = 4             # effective batch 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    eval_batch_size: int = 8        # RouteScorer batch. Kept at 8: *after* training
    #                                 the optimizer state is still resident, so eval
    #                                 has less free VRAM than a cold run (16 OOMs).

    # Paths
    adapter_dir: Path = field(default=HERE / "outputs" / "qlora-routing-adapter")
    figures_dir: Path = field(default=HERE / "figures")

    @property
    def n_labels(self) -> int:
        return len(ROUTES)


CONFIG = RoutingConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: Banking77 → routing queues
# ─────────────────────────────────────────────────────────────────────────────
def load_routing_dataset(cfg: RoutingConfig = CONFIG):
    """
    Returns a DatasetDict with 'train' / 'validation' / 'test', each having:
        text  (str)  – the customer complaint
        route (str)  – one of the 8 routing queues
        label (int)  – ROUTE2ID[route]
    Splits are stratified by route so rare queues stay represented.
    """
    from datasets import load_dataset, DatasetDict

    raw = load_dataset("PolyAI/banking77")
    intent_names = raw["train"].features["label"].names
    assert_full_coverage(intent_names)  # fail loudly if mapping drifts

    def remap(batch):
        routes = [route_for_intent(intent_names[i]) for i in batch["label"]]
        return {"route": routes, "label": [ROUTE2ID[r] for r in routes]}

    raw = raw.map(remap, batched=True, remove_columns=["label"])

    rng = np.random.default_rng(cfg.seed)

    def stratified_sample(ds, n):
        if n is None or n >= len(ds):
            return ds.shuffle(seed=cfg.seed)
        labels = np.array(ds["label"])
        per = max(1, n // len(ROUTES))
        idx = []
        for c in range(len(ROUTES)):
            pool = np.where(labels == c)[0]
            take = min(per, len(pool))
            idx.extend(rng.choice(pool, size=take, replace=False).tolist())
        rng.shuffle(idx)
        return ds.select(idx)

    train_full = raw["train"].shuffle(seed=cfg.seed)
    val = stratified_sample(train_full, cfg.val_size)
    val_ids = set(val["text"])
    train_remaining = train_full.filter(lambda r: r["text"] not in val_ids)
    train = stratified_sample(train_remaining, cfg.train_size)
    test = stratified_sample(raw["test"], cfg.test_size)

    return DatasetDict(train=train, validation=val, test=test)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────
def _route_rubric() -> str:
    lines = [f"- {r}: {ROUTE_DESCRIPTIONS[r]}" for r in ROUTES]
    return "\n".join(lines)


SYSTEM_PROMPT = (
    "You are a customer-support ticket router for a digital bank. "
    "Read the customer's message and assign it to exactly ONE routing queue "
    "from the list below. Reply with the queue name only — no explanation.\n\n"
    "Routing queues:\n" + _route_rubric()
)


def build_messages(text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'Customer message: "{text.strip()}"'},
    ]


def to_sft_dataset(ds, tokenizer):
    """
    Map a routing split to TRL's string prompt/completion format.

    We render the chat prompt to a *string* (add_generation_prompt=True) and keep
    the completion as the bare route name. With `completion_only_loss=True`, TRL
    tokenises prompt and completion separately and masks the prompt tokens, so the
    loss is computed only on the route name. (The messages/dict format does not
    mask reliably under the Qwen chat template — verified empirically.)
    """
    def fmt(row):
        prompt = tokenizer.apply_chat_template(
            build_messages(row["text"]), tokenize=False, add_generation_prompt=True,
        )
        return {"prompt": prompt, "completion": row["route"]}
    return ds.map(fmt, remove_columns=ds.column_names)


# ─────────────────────────────────────────────────────────────────────────────
# Constrained label scorer
# ─────────────────────────────────────────────────────────────────────────────
class RouteScorer:
    """
    Scores every routing queue as a candidate completion and returns
    length-normalised log-likelihoods → softmax probabilities over the 8 routes.
    """

    def __init__(self, model, tokenizer, cfg: RoutingConfig = CONFIG):
        self.model = model
        self.tok = tokenizer
        self.cfg = cfg
        self.routes = list(ROUTES)
        # Pre-tokenise prompt prefixes and route completions once.
        self._route_ids = [
            self.tok(r, add_special_tokens=False)["input_ids"] for r in self.routes
        ]

    def _prompt_ids(self, text: str) -> list[int]:
        out = self.tok.apply_chat_template(
            build_messages(text), add_generation_prompt=True, tokenize=True,
            return_dict=True,
        )
        return list(out["input_ids"])

    @torch.no_grad()
    def score_texts(self, texts: list[str]) -> np.ndarray:
        """Return (len(texts), 8) array of route probabilities."""
        device = next(self.model.parameters()).device
        pad_id = self.tok.pad_token_id or self.tok.eos_token_id

        # Build all (text, route) pairs.
        pairs, owners = [], []
        prompt_cache = [self._prompt_ids(t) for t in texts]
        for ti, p_ids in enumerate(prompt_cache):
            for ri, c_ids in enumerate(self._route_ids):
                pairs.append((p_ids, c_ids))
                owners.append((ti, ri))

        scores = np.full((len(texts), len(self.routes)), -1e9, dtype=np.float64)
        bs = self.cfg.eval_batch_size
        for start in range(0, len(pairs), bs):
            chunk = pairs[start : start + bs]
            full = [p + c for p, c in chunk]
            maxlen = max(len(x) for x in full)
            input_ids, attn, comp_mask = [], [], []
            for (p, c), seq in zip(chunk, full):
                pad = maxlen - len(seq)
                input_ids.append(seq + [pad_id] * pad)
                attn.append([1] * len(seq) + [0] * pad)
                m = [0] * len(p) + [1] * len(c) + [0] * pad  # mark completion tokens
                comp_mask.append(m)
            input_ids = torch.tensor(input_ids, device=device)
            attn = torch.tensor(attn, device=device)
            comp_mask = torch.tensor(comp_mask, device=device, dtype=torch.float32)

            logits = self.model(input_ids=input_ids, attention_mask=attn).logits
            # token t predicts token t+1 → align. Compute target log-probs as
            # (target_logit - logsumexp) WITHOUT a full-vocab float32 copy of the
            # logits — that copy is what OOMs an 8 GB card on a 152k vocab.
            logits = logits[:, :-1, :]
            tgt = input_ids[:, 1:]
            tgt_logit = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            lse = torch.logsumexp(logits, dim=-1)
            tok_logp = (tgt_logit - lse).float()
            del logits
            m = comp_mask[:, 1:]
            sum_lp = (tok_logp * m).sum(dim=1)
            n_tok = m.sum(dim=1).clamp(min=1)
            avg_lp = (sum_lp / n_tok).cpu().numpy()  # length-normalised
            for (ti, ri), v in zip([owners[start + j] for j in range(len(chunk))], avg_lp):
                scores[ti, ri] = v

        # softmax over routes → probabilities
        s = scores - scores.max(axis=1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(axis=1, keepdims=True)
        return p


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def routing_metrics(probs: np.ndarray, labels: np.ndarray, k: int = 3) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    preds = probs.argmax(axis=1)
    topk = np.argsort(-probs, axis=1)[:, :k]
    top_k_hit = np.mean([lab in row for lab, row in zip(labels, topk)])
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        f"top{k}_accuracy": float(top_k_hit),
    }
