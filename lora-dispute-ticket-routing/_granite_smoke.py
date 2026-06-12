import torch, numpy as np
from dataclasses import replace
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from routing_pipeline import CONFIG, load_routing_dataset, to_sft_dataset, RouteScorer, routing_metrics

MID = "ibm-granite/granite-4.1-3b"
CONFIG.train_size, CONFIG.val_size, CONFIG.test_size = 200, 16, 48
ds = load_routing_dataset(CONFIG)
tok = AutoTokenizer.from_pretrained(MID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
m = AutoModelForCausalLM.from_pretrained(MID, quantization_config=bnb, device_map="cuda")
m.config.use_cache = False
lora = LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
args = SFTConfig(output_dir="outputs/_gsmoke", max_length=384, per_device_train_batch_size=2,
    gradient_accumulation_steps=8, max_steps=10, logging_steps=5, learning_rate=2e-4, bf16=True,
    completion_only_loss=True, packing=False, optim="paged_adamw_8bit", gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}, save_strategy="no", report_to="none")
tr = SFTTrainer(model=m, args=args, train_dataset=to_sft_dataset(ds["train"], tok),
                processing_class=tok, peft_config=lora)
tr.model.print_trainable_parameters()
r = tr.train()
print("RESULT granite train_loss", round(r.training_loss, 4),
      "| peak_train_VRAM_GB", round(torch.cuda.max_memory_allocated()/1e9, 2))
import gc
ftm = tr.model.eval(); del tr; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
probs = RouteScorer(ftm, tok, replace(CONFIG, eval_batch_size=4)).score_texts(list(ds["test"]["text"]))
met = routing_metrics(probs, np.array(ds["test"]["label"]))
print("RESULT granite_FT_48ex", {k: round(v, 3) for k, v in met.items()})
print("RESULT eval_peak_VRAM_GB", round(torch.cuda.max_memory_allocated()/1e9, 2))
print("RESULT GRANITE_TRAIN_OK")
