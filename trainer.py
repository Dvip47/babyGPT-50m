# trainer.py
# ------------------------------------------------------------
import os, math, torch
from pathlib import Path
from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────
@dataclass
class TrainerConfig:
    # core
    max_iters: int
    batch_size: int
    block_size: int
    learning_rate: float
    lr_decay_iters: int
    warmup_iters: int
    min_lr: float
    device: str
    gradient_accumulation_steps: int
    # logging / ckpt
    out_dir: str
    eval_interval: int
    eval_iters: int
    log_interval: int
    dataset: str
    # optional
    weight_decay: float = 0.0
    adamw_beta2: float = 0.95
    grad_clip: float = 1.0
    keep_last_k: int = 5         # how many ckpts to keep (-1 = keep all)

# ─────────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, model, get_batch_fn, cfg: TrainerConfig):
        self.model      = model
        self.get_batch  = get_batch_fn
        self.cfg        = cfg
        self.device     = cfg.device
        self.iter_num   = 0
        # optimizer
        self.opt = torch.optim.AdamW(
            model.parameters(),
            lr           = cfg.learning_rate,
            betas        = (0.9, cfg.adamw_beta2),
            weight_decay = cfg.weight_decay,
        )
        # LR scheduler: linear warm‑up + cosine decay
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.opt,
            lr_lambda = self._lr_schedule
        )

    # ── learning‑rate schedule fn ────────────────────────────
    def _lr_schedule(self, step):
        cfg = self.cfg
        if step < cfg.warmup_iters:
            return step / max(1, cfg.warmup_iters)
        prog = (step - cfg.warmup_iters) / max(1, cfg.lr_decay_iters - cfg.warmup_iters)
        return max(cfg.min_lr / cfg.learning_rate, 0.5 * (1 + math.cos(math.pi * prog)))

    # ─────────────────────────────────────────────────────────
    def train_step(self):
        cfg, model = self.cfg, self.model
        model.train()

        total_loss = 0.0
        self.opt.zero_grad()

        for _ in range(cfg.gradient_accumulation_steps):
            x, y = self.get_batch('train', cfg)
            x, y = x.to(self.device), y.to(self.device)

            logits, loss = model(x, y)
            (loss / cfg.gradient_accumulation_steps).backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        self.opt.step()
        self.scheduler.step()

        self.iter_num += 1
        return total_loss / cfg.gradient_accumulation_steps

    # ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self):
        cfg, model = self.cfg, self.model
        model.eval()
        losses = []

        for _ in range(cfg.eval_iters):
            x, y = self.get_batch('val', cfg)
            x, y = x.to(self.device), y.to(self.device)
            _, loss = model(x, y)
            losses.append(loss.item())

        val_loss = sum(losses) / len(losses)
        self._save_checkpoint(val_loss)
        return val_loss

    # ─────────────────────────────────────────────────────────
    def _save_checkpoint(self, val_loss):
        cfg = self.cfg
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = out_dir / f"ckpt_{self.iter_num:07d}_{val_loss:.3f}.pt"
        torch.save(
            {
                "model"     : self.model.state_dict(),
                "model_args": self.model.config.__dict__,
                "iter_num"  : self.iter_num,
                "val_loss"  : val_loss,
            },
            ckpt_path,
        )

        # prune old ckpts if needed
        if cfg.keep_last_k > 0:
            ckpts = sorted(out_dir.glob("ckpt_*.pt"), key=os.path.getmtime)
            while len(ckpts) > cfg.keep_last_k:
                ckpts[0].unlink(); ckpts.pop(0)
