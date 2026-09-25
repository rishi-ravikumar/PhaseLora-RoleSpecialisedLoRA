import json
import pickle
import os
import random
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from tqdm import tqdm
import sys

# Usage:
# python phaseLora.py <model_name> <lora_r> <output_dir> <pred_output_file> <seed>
# Example:
# python phaseLora.py Qwen/Qwen2.5-3B-Instruct 4 phase_lora_ace05_qw3b_r4_s42 preds_phase_ace05_qw3b_r4_s42.pkl 42

MODEL_NAME       = sys.argv[1]
LORA_R           = int(sys.argv[2])
LORA_ALPHA       = float(2*LORA_R)
OUTPUT_DIR       = sys.argv[3]
PRED_OUTPUT_FILE = sys.argv[4]
SEED             = int(sys.argv[5])

TRAIN_FILE     = "train.jsonl"
TEST_FILE      = "test.jsonl"
LORA_DROPOUT   = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

EPOCHS         = 3
BATCH_SIZE     = 4
GRAD_ACCUM     = 4
LR             = 2e-4
WARMUP_RATIO   = 0.05
MAX_SEQ_LEN    = 512
DTYPE          = torch.bfloat16

MAX_NEW_TOKENS = 256
DO_SAMPLE      = False


# ─────────────────────────────────────────────
#  REPRODUCIBILITY
# ─────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
#  DUAL-ADAPTER LINEAR LAYER
# ─────────────────────────────────────────────

class DualLoRALinear(nn.Module):
    """
    Frozen base Linear + two LoRA adapter pairs (P and C).

    _seq_len controls which adapter(s) are used:
      _seq_len == 0  -> prefill mode -> lora_P only   (all prompt tokens)
      _seq_len == 1  -> decode mode  -> lora_C only   (one new token at a time)
      _seq_len >  1  -> train mode   -> blend via role_mask
      _seq_len == -1 -> not set      -> lora_C (safe fallback, should not occur)
    """

    def __init__(self, base_linear: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base    = base_linear
        in_f         = base_linear.in_features
        out_f        = base_linear.out_features
        self.scaling = alpha / r

        dev  = next(base_linear.parameters()).device
        dtyp = next(base_linear.parameters()).dtype

        self.lora_P_A = nn.Linear(in_f, r,     bias=False, device=dev, dtype=dtyp)
        self.lora_P_B = nn.Linear(r,    out_f, bias=False, device=dev, dtype=dtyp)
        self.lora_C_A = nn.Linear(in_f, r,     bias=False, device=dev, dtype=dtyp)
        self.lora_C_B = nn.Linear(r,    out_f, bias=False, device=dev, dtype=dtyp)
        self.dropout  = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_P_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_P_B.weight)
        nn.init.kaiming_uniform_(self.lora_C_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_C_B.weight)

        for p in self.base.parameters():
            p.requires_grad = False

        self.role_mask: Optional[torch.Tensor] = None
        self._seq_len:  int = -1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)

        if self._seq_len == 0:
            # Prefill: all prompt tokens -> lora_P
            delta = self.lora_P_B(self.dropout(self.lora_P_A(x))) * self.scaling

        elif self._seq_len == 1:
            # Decode: one new token at a time -> lora_C
            delta = self.lora_C_B(self.dropout(self.lora_C_A(x))) * self.scaling

        elif self._seq_len > 1 and self.role_mask is not None and x.shape[1] == self._seq_len:
            # Training: full [prompt + completion] sequence, blend via role_mask
            lora_P_out = self.lora_P_B(self.dropout(self.lora_P_A(x))) * self.scaling
            lora_C_out = self.lora_C_B(self.dropout(self.lora_C_A(x))) * self.scaling
            mask  = self.role_mask.unsqueeze(-1).to(x.dtype)   # (B, T, 1)
            delta = (1.0 - mask) * lora_P_out + mask * lora_C_out

        else:
            # Fallback — should not be reached in normal operation
            delta = self.lora_C_B(self.dropout(self.lora_C_A(x))) * self.scaling

        return base_out + delta


# ─────────────────────────────────────────────
#  DUAL-ADAPTER MODEL WRAPPER
# ─────────────────────────────────────────────

class DualLoRAModel(nn.Module):

    def __init__(self, base_model, target_modules, r, alpha, dropout):
        super().__init__()
        self.model = base_model
        self._dual_layers: list[DualLoRALinear] = []
        self._inject(target_modules, r, alpha, dropout)
        self._freeze_base()

    def _inject(self, target_modules, r, alpha, dropout):
        # Collect all replacements before mutating the module tree
        replacements = []
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(t in name for t in target_modules):
                continue
            replacements.append((name, module))

        for name, module in replacements:
            dual  = DualLoRALinear(module, r, alpha, dropout)
            self._dual_layers.append(dual)
            parts  = name.split(".")
            parent = self.model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], dual)

    def _freeze_base(self):
        for name, param in self.named_parameters():
            param.requires_grad = ("lora_P" in name or "lora_C" in name)

    def _set_mode(self, mode: str, role_mask: Optional[torch.Tensor] = None,
                  seq_len: int = -1):
        for layer in self._dual_layers:
            if mode == "train":
                layer.role_mask = role_mask
                layer._seq_len  = seq_len
            elif mode == "prefill":
                layer.role_mask = None
                layer._seq_len  = 0
            elif mode == "decode":
                layer.role_mask = None
                layer._seq_len  = 1

    def _clear(self):
        for layer in self._dual_layers:
            layer.role_mask = None
            layer._seq_len  = -1

    # ── training forward ─────────────────────────────────────────────────────

    def forward(self, input_ids, attention_mask=None, labels=None,
                role_mask=None, **kwargs):
        if role_mask is not None:
            self._set_mode("train", role_mask=role_mask,
                           seq_len=input_ids.shape[1])
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         labels=labels, **kwargs)
        self._clear()
        return out

    # ── two-phase inference ───────────────────────────────────────────────────

    def generate_dual(self, input_ids, attention_mask, max_new_tokens,
                      pad_token_id, eos_token_id):
        """
        Phase 1 - Prefill (lora_P):
            Run the full prompt through the model to populate the KV cache.
            No token is sampled here — purely for the cache.

        Phase 2 - Decode (lora_C):
            Switch adapter to lora_C, then generate all new tokens one at a
            time using the cached prompt representations. The first generated
            token is produced here, under lora_C — matching the training split.
        """
        device = input_ids.device

        # Phase 1: prefill with lora_P — build KV cache, sample nothing
        self._set_mode("prefill")
        with torch.no_grad():
            prefill_out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
        past_kv = prefill_out.past_key_values

        # Phase 2: decode with lora_C — generate all tokens including the first
        self._set_mode("decode")

        next_token   = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated    = [next_token]
        current_attn = torch.cat(
            [attention_mask,
             torch.ones(input_ids.shape[0], 1, device=device, dtype=attention_mask.dtype)],
            dim=1
        )

        with torch.no_grad():
            for _ in range(max_new_tokens - 1):
                out = self.model(
                    input_ids=next_token,
                    attention_mask=current_attn,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                past_kv    = out.past_key_values
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated.append(next_token)

                current_attn = torch.cat(
                    [current_attn,
                     torch.ones(input_ids.shape[0], 1,
                                device=device, dtype=attention_mask.dtype)],
                    dim=1
                )

                if (next_token == eos_token_id).all():
                    break

        self._clear()
        return torch.cat(generated, dim=1)   # (B, gen_len)

    # ── persistence ──────────────────────────────────────────────────────────

    def save_adapters(self, path: str):
        os.makedirs(path, exist_ok=True)
        state = {k: v for k, v in self.state_dict().items()
                 if "lora_P" in k or "lora_C" in k}
        torch.save(state, os.path.join(path, "dual_lora_adapters.pt"))
        print(f"Saved {len(state)} adapter tensors -> {path}/dual_lora_adapters.pt")

    def load_adapters(self, path: str):
        state  = torch.load(os.path.join(path, "dual_lora_adapters.pt"),
                            map_location="cpu")
        missing, _ = self.load_state_dict(state, strict=False)
        lora_missing = [k for k in missing if "lora_P" in k or "lora_C" in k]
        print(f"Loaded {len(state)} adapter tensors. "
              f"LoRA keys missing (should be 0): {len(lora_missing)}")

    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────

class DualLoRADataset(Dataset):
    def __init__(self, filepath: str, tokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.samples   = []
        with open(filepath) as f:
            for line in f:
                obj = json.loads(line.strip())
                self.samples.append((obj["prompt"], obj["completion"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, completion = self.samples[idx]

        messages    = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        full_enc  = self.tokenizer(
            prompt_text + completion,
            max_length=self.max_len,
            truncation=True,
            add_special_tokens=False,
        )
        input_ids  = full_enc["input_ids"]
        attn_mask  = full_enc["attention_mask"]
        prompt_len = min(len(prompt_ids), len(input_ids))

        labels = [-100] * prompt_len + input_ids[prompt_len:]

        if input_ids[-1] != self.tokenizer.eos_token_id:
            if len(input_ids) < self.max_len:
                input_ids.append(self.tokenizer.eos_token_id)
                attn_mask.append(1)
                labels.append(self.tokenizer.eos_token_id)

        labels    = labels[:len(input_ids)]
        role_mask = [0] * prompt_len + [1] * (len(input_ids) - prompt_len)
        role_mask = role_mask[:len(input_ids)]

        return {
            "input_ids":      torch.tensor(input_ids,  dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask,  dtype=torch.long),
            "labels":         torch.tensor(labels,     dtype=torch.long),
            "role_mask":      torch.tensor(role_mask,  dtype=torch.bool),
        }


def collate_fn(batch, pad_id):
    max_len   = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros(len(batch), max_len,          dtype=torch.long)
    labels    = torch.full((len(batch), max_len), -100,   dtype=torch.long)
    role_mask = torch.zeros(len(batch), max_len,          dtype=torch.bool)

    for i, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        input_ids[i, :n] = b["input_ids"]
        attn_mask[i, :n] = b["attention_mask"]
        labels[i, :n]    = b["labels"]
        role_mask[i, :n] = b["role_mask"]

    return {"input_ids": input_ids, "attention_mask": attn_mask,
            "labels": labels, "role_mask": role_mask}


# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────

def train():
    set_seed(SEED)
    print(f"Seed set to {SEED}")

    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map="auto", trust_remote_code=True
    )

    print("Injecting dual LoRA adapters...")
    model  = DualLoRAModel(base_model, TARGET_MODULES, LORA_R, LORA_ALPHA, LORA_DROPOUT)
    total  = model.total_parameters()
    train_ = model.trainable_parameters()
    print(f"Total: {total/1e6:.1f}M | Trainable: {train_/1e6:.1f}M ({100*train_/total:.2f}%)")
    print(f"Dual layers injected: {len(model._dual_layers)}")

    device  = next(model.parameters()).device
    dataset = DualLoRADataset(TRAIN_FILE, tokenizer, MAX_SEQ_LEN)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))

    optimizer    = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01
    )
    total_steps  = (len(loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"Training {EPOCHS} epochs | {total_steps} optimizer steps")
    model.train()

    # Loss tracking: per-step (every optimizer step) and per-epoch average
    epoch_losses      = []
    step_losses       = []   # each entry: {"global_step": int, "epoch": int, "loss": float}
    global_step       = 0
    total_loss_global = 0.0  # accumulated across all epochs for smooth global curve
    total_batches     = 0    # total batch count across all epochs

    for epoch in range(EPOCHS):
        total_loss        = 0.0  # epoch-local, for epoch_losses
        total_steps_epoch = 0
        optimizer.zero_grad()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["labels"].to(device)
            role_mask = batch["role_mask"].to(device)

            out  = model(input_ids=input_ids, attention_mask=attn_mask,
                         labels=labels, role_mask=role_mask)
            loss = out.loss / GRAD_ACCUM
            loss.backward()
            total_loss        += out.loss.item()   # epoch-local accumulation (unscaled)
            total_loss_global += out.loss.item()   # global accumulation across epochs
            total_steps_epoch += 1
            total_batches     += 1

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Global running average across all epochs — smooth, no resets
                running_avg = total_loss_global / total_batches
                step_losses.append({
                    "global_step": global_step,
                    "epoch":       epoch + 1,
                    "loss":        running_avg,
                })

                pbar.set_postfix(loss=f"{running_avg:.4f}",
                                 lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = total_loss / total_steps_epoch
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    model.save_adapters(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # ── Save run metadata + loss curves ─────────────────────────────────────
    run_info = {
        "model":        MODEL_NAME,
        "method":       "phase_lora",
        "lora_r":       LORA_R,
        "lora_alpha":   LORA_ALPHA,
        "seed":         SEED,
        "output_dir":   OUTPUT_DIR,
        "epoch_losses": epoch_losses,       # list of length EPOCHS — avg loss per epoch
        "step_losses":  step_losses,        # list of dicts: {global_step, epoch, loss}
        "epochs":       EPOCHS,
        "lr":           LR,
        "batch_size":   BATCH_SIZE,
        "grad_accum":   GRAD_ACCUM,
        "trainable_params": train_,
        "total_params":     total,
    }
    meta_path = os.path.join(OUTPUT_DIR, "run_info.json")
    with open(meta_path, "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"Saved run metadata -> {meta_path}")
    print("Training complete.")


# ─────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────

def infer():
    print("Loading tokenizer and base model for inference...")
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map="auto", trust_remote_code=True
    )

    model = DualLoRAModel(base_model, TARGET_MODULES, LORA_R, LORA_ALPHA, LORA_DROPOUT)
    model.load_adapters(OUTPUT_DIR)
    model.eval()

    device = next(model.parameters()).device

    with open(TEST_FILE) as f:
        lines = [json.loads(l.strip()) for l in f]

    results = []
    print(f"Generating on {len(lines)} test examples...")
    for i, obj in tqdm(enumerate(lines)):
        prompt      = obj["prompt"]
        messages    = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(prompt_text, return_tensors="pt",
                        add_special_tokens=False).to(device)

        gen_ids = model.generate_dual(
            input_ids      = enc["input_ids"],
            attention_mask = enc["attention_mask"],
            max_new_tokens = MAX_NEW_TOKENS,
            pad_token_id   = tokenizer.pad_token_id,
            eos_token_id   = tokenizer.eos_token_id,
        )

        prediction = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        if i < 5:
            print(prediction)
        results.append(prediction)
        if i >= 499:
            break

    with open(PRED_OUTPUT_FILE, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved {len(results)} predictions -> {PRED_OUTPUT_FILE}")


train()
infer()