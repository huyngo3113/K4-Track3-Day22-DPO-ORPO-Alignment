# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB3 — DPO Training (the main event)
#
# **Stack:** TRL `DPOTrainer` + `DPOConfig(beta=0.1, lr=5e-7)` from deck §5.2.
# Maps to deck §3 (DPO derivation), §3.4 (failure modes — read closely!), §5.2 (TRL impl).
#
# > **Mục tiêu:** train DPO adapter on top of NB1 SFT-mini. Plot reward curves
# > (cả `chosen_rewards` và `rejected_rewards`). Save adapter to `adapters/dpo/`.
# >
# > Đây là **the** notebook quan trọng nhất của lab — 25/100 pts đến từ đây.
# > Đặc biệt là: **plot cả 2 curve riêng biệt**, không chỉ reward gap (deck §3.4).

# %% [markdown]
# ## 0. Setup

# %%
import os
from pathlib import Path

COMPUTE_TIER = os.environ.get("COMPUTE_TIER", "T4").upper()

if COMPUTE_TIER == "T4":
    BASE_MODEL = "unsloth/Qwen2.5-3B-bnb-4bit"
    INSTRUCT_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # chat_template source on the no-Unsloth fallback path
    MAX_LEN = 512
    MAX_PROMPT_LEN = 256
    PER_DEVICE_BATCH = 1
    GRAD_ACCUM = 8
else:
    BASE_MODEL = "unsloth/Qwen2.5-7B-bnb-4bit"
    INSTRUCT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    MAX_LEN = 1024
    MAX_PROMPT_LEN = 512
    PER_DEVICE_BATCH = 1
    GRAD_ACCUM = 4

# Hyperparameters from deck §5.2 (TRL DPOTrainer implementation frame)
BETA = float(os.environ.get("DPO_BETA", "0.1"))
LR = float(os.environ.get("DPO_LR", "5e-7"))
EPOCHS = int(os.environ.get("DPO_EPOCHS", "1"))

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
SFT_PATH = REPO_ROOT / "adapters" / "sft-mini"
DPO_OUT = REPO_ROOT / "adapters" / "dpo"
PREF_PATH = REPO_ROOT / "data" / "pref" / "train.parquet"

DPO_OUT.mkdir(parents=True, exist_ok=True)

assert SFT_PATH.exists(), f"NB1 must run first — {SFT_PATH} missing"
assert PREF_PATH.exists(), f"NB2 must run first — {PREF_PATH} missing"

print(f"COMPUTE_TIER:    {COMPUTE_TIER}")
print(f"BASE_MODEL:      {BASE_MODEL}")
print(f"DPO hyperparams: beta={BETA}  lr={LR}  epochs={EPOCHS}")
print(f"max_length:      {MAX_LEN}  (prompt={MAX_PROMPT_LEN})")
print(f"effective batch: {PER_DEVICE_BATCH * GRAD_ACCUM}")
print(f"SFT input:       {SFT_PATH}")
print(f"output:          {DPO_OUT}")

# %%
import torch

assert torch.cuda.is_available(), "DPO needs a CUDA GPU. See HARDWARE-GUIDE.md."

# %% [markdown]
# ## 1. Load policy + reference (the VRAM story)
#
# **Critical:** DPO scores each answer under the policy (trainable) AND a frozen
# reference. With PEFT we do **not** load a second model -- TRL toggles the LoRA
# adapter *off* to get the reference forward pass on the same 4-bit base. The
# extra VRAM vs SFT comes from two forward passes + holding chosen AND rejected
# sequences, not from a second copy of the weights.

# %%
# xformers' Triton GQA kernel (what Unsloth's fast attention calls for
# Qwen2.5) requires compute capability >= 8.0 (Ampere+). DPOTrainer is the
# first place that hits it because its backward pass runs on the
# concatenated chosen+rejected batch -- NB1/SFT never triggers a backward
# through this kernel.
#
# **This is a process-wide monkeypatch, not a per-model setting.** `import
# unsloth` (even indirectly, e.g. `from unsloth.chat_templates import ...`)
# patches `Qwen2Attention.forward` at the class level for the whole Python
# process -- so a plain `transformers.AutoModelForCausalLM` load is
# affected too if Unsloth was ever imported earlier in this kernel (e.g. by
# running NB1 in the same session). Passing `attn_implementation="eager"`
# does NOT override this, because the patch replaces the method, not the
# config-driven dispatch.
#
# **You must restart the runtime AFTER NB1 and BEFORE this cell** so this
# kernel has never imported unsloth. NB1's adapter is already saved to
# disk, so you don't need to re-run its training cell -- NB2 doesn't
# import unsloth either (it loads the tokenizer from the saved adapter
# dir), so re-running NB2 then this cell in the fresh kernel is enough.
if COMPUTE_TIER == "T4":
    import sys
    assert "unsloth" not in sys.modules, (
        "unsloth is already imported in this kernel (likely from running NB1 in the "
        "same session). Its attention patch is process-wide and attn_implementation="
        "\"eager\" cannot override it. Restart the runtime, then re-run from NB2 -- "
        "NB1's adapter is already saved to disk, no need to retrain it."
    )

from peft import PeftModel, LoraConfig, get_peft_model

LORA_KWARGS = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

major, minor = torch.cuda.get_device_capability()
if (major, minor) < (8, 0):
    print(f"GPU capability {major}.{minor} < 8.0 -- loading via plain transformers (eager attention), never importing unsloth.")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # unsloth.chat_templates.get_chat_template() would re-trigger the global
    # patch we're avoiding -- copy the chat_template from the Instruct
    # sibling instead (same family, same template, zero unsloth import).
    tokenizer.chat_template = AutoTokenizer.from_pretrained(INSTRUCT_MODEL).chat_template
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.float16, attn_implementation="eager", device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base, str(SFT_PATH), is_trainable=True)
    print(f"Policy: {model.__class__.__name__} with SFT adapter loaded")

    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA_KWARGS))
    model.gradient_checkpointing_enable()
else:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    # Policy — gets new DPO LoRA adapter on top of SFT LoRA
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load SFT adapter on top of base
    model = PeftModel.from_pretrained(model, str(SFT_PATH), is_trainable=True)
    print(f"Policy: {model.__class__.__name__} with SFT adapter loaded")

    # Wrap policy with NEW LoRA adapter for DPO updates (don't merge SFT — keep stacked)
    # Unsloth re-applies LoRA on top of the existing PeftModel.
    model = FastLanguageModel.get_peft_model(
        model,
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
        **LORA_KWARGS,
    )
print(f"Trainable params (DPO LoRA): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# %% [markdown]
# > **Why no separate `ref_model=` argument?** Modern TRL (≥ 0.12) auto-detects
# > PEFT models and uses the *base model without the adapter* as the reference.
# > That's the same memory layout: 1 base + 2 adapter sets in VRAM. No deepcopy
# > needed.

# %% [markdown]
# ## 2. Build DPOConfig (deck §5.2 hyperparameters)

# %%
from trl import DPOConfig

dpo_config = DPOConfig(
    output_dir=str(DPO_OUT.parent / "dpo-checkpoints"),
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    beta=BETA,
    max_length=MAX_LEN,
    max_prompt_length=MAX_PROMPT_LEN,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="no",
    optim="adamw_8bit",
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    seed=42,
    loss_type="sigmoid",         # DPO standard (alternatives: ipo, hinge, kto)
    report_to="none",
)

print(f"DPOConfig: beta={dpo_config.beta}  lr={dpo_config.learning_rate}  loss_type={dpo_config.loss_type}")

# %% [markdown]
# ## 3. Load preference data

# %%
from datasets import Dataset

pref_ds = Dataset.from_parquet(str(PREF_PATH))
print(f"Loaded {len(pref_ds)} preference pairs from {PREF_PATH}")
print(f"Columns: {pref_ds.column_names}")

# %% [markdown]
# ## 4. Train

# %%
from trl import DPOTrainer

trainer = DPOTrainer(
    model=model,
    ref_model=None,                # auto-derived from PEFT base
    args=dpo_config,
    train_dataset=pref_ds,
    processing_class=tokenizer,
)

# %%
train_result = trainer.train()
print(f"\nFinal DPO loss: {train_result.training_loss:.4f}")

# %% [markdown]
# ## 5. Plot reward curves — THE diagnostic
#
# **Read deck §3.4 before interpreting these.** A growing reward gap can come from:
# - **(intended)** chosen reward going up + rejected staying flat
# - **(intended)** chosen rising slowly + rejected falling fast
# - **(likelihood displacement)** chosen reward going *down* + rejected falling faster
#
# The third case is what Razin et al. 2024 documented. It's not a bug, but it
# tells you the model is finding a way to widen the gap that doesn't necessarily
# improve actual chosen probability.

# %%
import matplotlib.pyplot as plt
import pandas as pd

logs = pd.DataFrame(trainer.state.log_history)
logs = logs[logs["loss"].notna() if "loss" in logs.columns else logs.index].copy()

# TRL DPO logs include rewards/chosen, rewards/rejected, rewards/margins, kl
chosen_col = "rewards/chosen" if "rewards/chosen" in logs.columns else None
rejected_col = "rewards/rejected" if "rewards/rejected" in logs.columns else None

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

if chosen_col and rejected_col:
    axes[0].plot(logs["step"], logs[chosen_col], label="chosen reward", color="#2e548a", linewidth=1.5)
    axes[0].plot(logs["step"], logs[rejected_col], label="rejected reward", color="#c83538", linewidth=1.5)
    axes[0].axhline(0, color="#888", linestyle=":", linewidth=0.7)
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Implicit reward (log π/π_ref)")
    axes[0].set_title("Chosen vs Rejected rewards")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    gap = logs[chosen_col] - logs[rejected_col]
    axes[1].plot(logs["step"], gap, color="#1a3355", linewidth=1.8)
    axes[1].axhline(0, color="#888", linestyle=":", linewidth=0.7)
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Reward gap (chosen − rejected)")
    axes[1].set_title("Reward gap (the headline number)")
    axes[1].grid(True, alpha=0.3)
else:
    axes[0].text(0.5, 0.5, "No reward columns in trainer.state.log_history.\nLikely TRL version mismatch.",
                 ha="center", va="center", transform=axes[0].transAxes)
    axes[1].text(0.5, 0.5, "—", ha="center", va="center", transform=axes[1].transAxes)

fig.suptitle(f"DPO reward curves · {COMPUTE_TIER} · β={BETA} · lr={LR}", y=1.02)
fig.tight_layout()

screenshot_dir = REPO_ROOT / "submission" / "screenshots"
screenshot_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(screenshot_dir / "03-dpo-reward-curves.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 5a. Failure-mode self-check
#
# Read this cell carefully — it tells you which kind of "reward gap up" you got.

# %%
if chosen_col and rejected_col and len(logs) >= 5:
    last_chosen = logs[chosen_col].iloc[-5:].mean()
    last_rejected = logs[rejected_col].iloc[-5:].mean()
    last_gap = last_chosen - last_rejected
    first_chosen = logs[chosen_col].iloc[:5].mean()

    chosen_delta = last_chosen - first_chosen

    print(f"END  chosen reward:    {last_chosen:+.3f}")
    print(f"END  rejected reward:  {last_rejected:+.3f}")
    print(f"END  reward gap:       {last_gap:+.3f}")
    print()

    if last_gap < 0:
        print("✗ FAILURE: reward gap went NEGATIVE. DPO did the opposite of what you wanted.")
        print("  Likely causes: data quality (chosen/rejected swapped?), beta too high, lr too low.")
    elif chosen_delta < -0.5 and last_gap > 0:
        print("⚠  LIKELIHOOD DISPLACEMENT (deck §3.4):")
        print(f"   Reward gap is positive ({last_gap:+.3f}) — good!")
        print(f"   But chosen reward FELL by {chosen_delta:+.3f} during training.")
        print("   The gap grew because rejected fell faster than chosen.")
        print("   Document this in REFLECTION § 3 — it's a teachable moment, not a bug.")
    elif chosen_delta > 0 and last_gap > 0:
        print("✓ INTENDED: chosen reward UP and gap positive. Classic DPO success.")
    else:
        print("?  AMBIGUOUS: weak chosen movement + positive gap. Try longer training or higher lr.")

# %% [markdown]
# ## 6. Save adapter

# %%
trainer.model.save_pretrained(str(DPO_OUT))
tokenizer.save_pretrained(str(DPO_OUT))
print(f"Saved DPO adapter to {DPO_OUT}")

# Save the headline metrics for verify.py + REFLECTION
import json

metrics = {
    "compute_tier": COMPUTE_TIER,
    "base_model": BASE_MODEL,
    "beta": BETA,
    "lr": LR,
    "epochs": EPOCHS,
    "final_train_loss": float(train_result.training_loss),
    "end_chosen_reward": float(last_chosen) if chosen_col else None,
    "end_rejected_reward": float(last_rejected) if rejected_col else None,
    "end_reward_gap": float(last_gap) if chosen_col and rejected_col else None,
}
(DPO_OUT / "dpo_metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"Wrote metrics to {DPO_OUT / 'dpo_metrics.json'}")

# %% [markdown]
# ## 7. Vibe-coding callout
#
# Now's the time for the **β experiment** if you want the +6 rigor add-on.
#
# `make beta-sweep` runs this notebook 3 times with `DPO_BETA ∈ {0.05, 0.1, 0.5}`
# and saves to `adapters/dpo-b{0.05,0.1,0.5}/`. Plot the results yourself:
#
# ```python
# import json
# import matplotlib.pyplot as plt
# from pathlib import Path
#
# results = []
# for d in sorted((REPO_ROOT / "adapters").glob("dpo-b*")):
#     m = json.loads((d / "dpo_metrics.json").read_text())
#     results.append((m["beta"], m["end_reward_gap"]))
# # plot β vs reward_gap
# ```
#
# **Think-hard zone:** what's the *expected* shape of the β-vs-reward-gap curve?
# Hypothesize before you look at the data. (Hint: deck §3.3.)
#
# **Next:** NB4 — qualitative side-by-side comparison.
