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
    # python standardLora.py <model_name> <lora_r> <output_dir> <pred_output_file> <seed>
    # Example:
    # python standardLora.py Qwen/Qwen2.5-3B-Instruct 4 single_lora_ace05_qw3b_r4_s42 preds_single_ace05_qw3b_r4_s42.pkl 42

    MODEL_NAME       = "mistralai/Mistral-7B-Instruct-v0.3"
    LORA_R           = 16
    LORA_ALPHA       = 16
    OUTPUT_DIR       = "outputs"
    PRED_OUTPUT_FILE = "erfgc_m7b_3eps.pkl"
    SEED             = 314

    TRAIN_FILE       = "train.jsonl"
    TEST_FILE        = "test.jsonl"
    LORA_DROPOUT     = 0.05
    TARGET_MODULES   = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]

    EPOCHS           = 3
    BATCH_SIZE       = 4
    GRAD_ACCUM       = 4
    LR               = 2e-4
    WARMUP_RATIO     = 0.05
    MAX_SEQ_LEN      = 2048
    DTYPE            = torch.bfloat16

    MAX_NEW_TOKENS   = 2048
    DO_SAMPLE        = False


    # ─────────────────────────────────────────────
    #  REPRODUCIBILITY
    # ─────────────────────────────────────────────

    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Makes convolutions deterministic at slight perf cost — fine for LoRA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    # ─────────────────────────────────────────────
    #  SINGLE-ADAPTER MODULE
    # ─────────────────────────────────────────────

    class SingleLoRALinear(nn.Module):
        """
        Wraps a frozen Linear layer with one standard LoRA adapter.
        Applied uniformly to all token positions (no role distinction).
        """

        def __init__(self, base_linear: nn.Linear, r: int, alpha: float, dropout: float):
            super().__init__()
            self.base    = base_linear
            in_f         = base_linear.in_features
            out_f        = base_linear.out_features
            self.scaling = alpha / r

            base_device = next(base_linear.parameters()).device
            base_dtype  = next(base_linear.parameters()).dtype

            self.lora_A  = nn.Linear(in_f, r,     bias=False, device=base_device, dtype=base_dtype)
            self.lora_B  = nn.Linear(r,    out_f, bias=False, device=base_device, dtype=base_dtype)
            self.dropout = nn.Dropout(dropout)

            nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
            nn.init.zeros_(self.lora_B.weight)

            for p in self.base.parameters():
                p.requires_grad = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.base(x) + self.lora_B(self.dropout(self.lora_A(x))) * self.scaling


    class SingleLoRAModel(nn.Module):
        def __init__(self, base_model, target_modules, r, alpha, dropout):
            super().__init__()
            self.model = base_model
            self._inject(target_modules, r, alpha, dropout)
            self._freeze_base()

        def _inject(self, target_modules, r, alpha, dropout):
            for name, module in self.model.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                if not any(t in name for t in target_modules):
                    continue
                single = SingleLoRALinear(module, r, alpha, dropout)
                parts  = name.split(".")
                parent = self.model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], single)

        def _freeze_base(self):
            for name, param in self.named_parameters():
                param.requires_grad = ("lora_A" in name or "lora_B" in name)

        def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
            kwargs.pop("role_mask", None)
            return self.model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=labels, **kwargs)

        def generate(self, input_ids, attention_mask=None, **kwargs):
            return self.model.generate(input_ids=input_ids,
                                    attention_mask=attention_mask, **kwargs)

        def save_adapters(self, path: str):
            os.makedirs(path, exist_ok=True)
            state = {k: v for k, v in self.state_dict().items()
                    if "lora_A" in k or "lora_B" in k}
            torch.save(state, os.path.join(path, "single_lora_adapters.pt"))
            print(f"Saved adapters -> {path}/single_lora_adapters.pt")

        def load_adapters(self, path: str):
            state = torch.load(os.path.join(path, "single_lora_adapters.pt"),
                            map_location="cpu")
            missing, unexpected = self.load_state_dict(state, strict=False)
            print(f"Loaded adapters. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

        def trainable_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def total_parameters(self):
            return sum(p.numel() for p in self.parameters())


    # ─────────────────────────────────────────────
    #  DATASET
    # ─────────────────────────────────────────────

    class SingleLoRADataset(Dataset):
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

            messages     = [{"role": "user", "content": prompt}]
            prompt_text  = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids   = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

            full_text    = prompt_text + completion
            full_enc     = self.tokenizer(full_text, max_length=self.max_len,
                                        truncation=True, add_special_tokens=False)
            input_ids    = full_enc["input_ids"]
            attn_mask    = full_enc["attention_mask"]
            prompt_len   = min(len(prompt_ids), len(input_ids))

            labels = [-100] * prompt_len + input_ids[prompt_len:]

            if input_ids[-1] != self.tokenizer.eos_token_id:
                if len(input_ids) < self.max_len:
                    input_ids.append(self.tokenizer.eos_token_id)
                    attn_mask.append(1)
                    labels.append(self.tokenizer.eos_token_id)

            labels = labels[:len(input_ids)]

            return {
                "input_ids":      torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
                "labels":         torch.tensor(labels,    dtype=torch.long),
            }


    def collate_fn(batch, pad_id):
        max_len   = max(b["input_ids"].shape[0] for b in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn_mask = torch.zeros(len(batch), max_len,          dtype=torch.long)
        labels    = torch.full((len(batch), max_len), -100,   dtype=torch.long)

        for i, b in enumerate(batch):
            n = b["input_ids"].shape[0]
            input_ids[i, :n] = b["input_ids"]
            attn_mask[i, :n] = b["attention_mask"]
            labels[i, :n]    = b["labels"]

        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}


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

        print("Injecting single LoRA adapter...")
        model = SingleLoRAModel(
            base_model, target_modules=TARGET_MODULES,
            r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT
        )

        total  = model.total_parameters()
        train_ = model.trainable_parameters()
        print(f"Total params: {total/1e6:.1f}M | Trainable: {train_/1e6:.1f}M "
            f"({100*train_/total:.2f}%)")

        device = next(model.parameters()).device

        dataset = SingleLoRADataset(TRAIN_FILE, tokenizer, MAX_SEQ_LEN)
        loader  = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id)
        )

        optimizer    = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01
        )
        total_steps  = (len(loader) // GRAD_ACCUM) * EPOCHS
        warmup_steps = int(total_steps * WARMUP_RATIO)
        scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        print(f"Training for {EPOCHS} epochs | {total_steps} optimizer steps")
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

                out  = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                loss = out.loss / GRAD_ACCUM
                loss.backward()
                total_loss        += out.loss.item()   # epoch-local accumulation
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

        # ── Save run metadata + loss curve ──────────────────────────────────────
        run_info = {
            "model":        MODEL_NAME,
            "method":       "single_lora",
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

        model = SingleLoRAModel(
            base_model, target_modules=TARGET_MODULES,
            r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT
        )
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

            with torch.no_grad():
                out_ids = model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=DO_SAMPLE,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            gen_ids    = out_ids[0][enc["input_ids"].shape[1]:]
            prediction = tokenizer.decode(gen_ids, skip_special_tokens=True)

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