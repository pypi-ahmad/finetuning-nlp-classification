"""
triage_worker.py
─────────────────────────────────────────────────────────────────────────────
Train + evaluate ONE backbone in an isolated subprocess, then exit so the OS
reclaims 100% of its GPU memory. The notebook's active-learning loop (6×) and
the Part-B backbone bake-off (Granite, Phi) call this once per model — that way
VRAM never accumulates across repeated trainings (HF Trainer + accelerate +
bitsandbytes don't fully release in-process; ~0.5 GB leaks per fine-tune).

Usage:  python triage_worker.py <config.json>

config.json keys:
  model_id, target_modules (list), train_parquet (rows to train on),
  bs, accum, epochs, eval_bs, train_cap (to reproduce deterministic splits),
  zero_shot (bool), score_pool (bool), pool_sample (int|null),
  out_dir, out_json
Writes out_json: {zero_shot?, fine_tuned, peak_gb, pool_idx?, pool_l1_probs?}.
"""
import sys, json, os
from dataclasses import replace

import numpy as np
import pandas as pd
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          set_seed)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

from triage_config import CONFIG
from triage_data import build_dataframe, make_splits
from triage_model import make_sft_records, HierScorer, hier_evaluate


def main(cfg_path):
    c = json.load(open(cfg_path))
    set_seed(CONFIG.seed)
    CONFIG.train_cap = c["train_cap"]                 # reproduce identical splits
    df, hier = build_dataframe(CONFIG)
    sp = make_splits(df, CONFIG)
    l1_labels = sorted(hier.keys())
    test_df = sp["test"]
    if c.get("test_sample"):
        test_df = test_df.sample(min(c["test_sample"], len(test_df)),
                                 random_state=CONFIG.seed).reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(c["model_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eval_cfg = replace(CONFIG, eval_batch_size=c["eval_bs"])

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    mdl = AutoModelForCausalLM.from_pretrained(c["model_id"], quantization_config=bnb,
                                               device_map="cuda")
    out = {}
    if c.get("zero_shot"):
        zs, _ = hier_evaluate(HierScorer(mdl, tok, eval_cfg), test_df, hier)
        out["zero_shot"] = zs

    train_df = pd.read_parquet(c["train_parquet"])
    mdl.config.use_cache = False
    args = SFTConfig(output_dir=c["out_dir"], num_train_epochs=c["epochs"],
        per_device_train_batch_size=c["bs"], gradient_accumulation_steps=c["accum"],
        learning_rate=CONFIG.learning_rate, warmup_ratio=CONFIG.warmup_ratio,
        lr_scheduler_type="cosine", max_length=CONFIG.max_length,
        completion_only_loss=True, packing=False, bf16=True, optim="paged_adamw_8bit",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=50, save_strategy="no", report_to="none")
    torch.cuda.reset_peak_memory_stats()
    tr = SFTTrainer(model=mdl, args=args,
                    train_dataset=make_sft_records(train_df, hier, tok, CONFIG),
                    processing_class=tok, peft_config=LoraConfig(task_type="CAUSAL_LM",
                        r=CONFIG.lora_r, lora_alpha=CONFIG.lora_alpha,
                        lora_dropout=CONFIG.lora_dropout, target_modules=list(c["target_modules"])))
    tr.train()
    sc = HierScorer(tr.model.eval(), tok, eval_cfg)
    ft, _ = hier_evaluate(sc, test_df, hier)
    out["fine_tuned"] = ft
    out["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    if c.get("score_pool"):
        pool = sp["pool"]
        if c.get("pool_sample"):
            pool = pool.sample(min(c["pool_sample"], len(pool)),
                               random_state=CONFIG.seed)
        probs = sc.score_l1(list(pool["text"]), l1_labels)
        out["pool_idx"] = [int(i) for i in pool.index.tolist()]
        out["pool_l1_probs"] = [[float(d[l]) for l in l1_labels] for d in probs]

    json.dump(out, open(c["out_json"], "w"))
    print("WORKER_DONE", c["model_id"], "peak", out["peak_gb"], "GB")


if __name__ == "__main__":
    main(sys.argv[1])
