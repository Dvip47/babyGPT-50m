import torch, os

class TrainerConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Trainer:
    def __init__(self, model, get_batch_fn, cfg: TrainerConfig):
        self.model = model
        self.get_batch = get_batch_fn
        self.cfg = cfg
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
        self.device = cfg.device

    def train(self):
        model, cfg = self.model, self.cfg
        model.train()

        for it in range(cfg.max_iters):
            x, y = self.get_batch('train', cfg)
            logits, loss = model(x.to(self.device), y.to(self.device))
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            self.opt.step()

            if it % cfg.log_interval == 0:
                with torch.no_grad():
                    vx, vy = self.get_batch('val', cfg)
                    vloss = model(vx.to(self.device), vy.to(self.device))[1]
                print(f"iter {it}: train loss {loss.item():.3f}, val loss {vloss.item():.3f}")

            if it % cfg.eval_interval == 0 and it > 0:
                os.makedirs(cfg.out_dir, exist_ok=True)
                ckpt = {
                    'model': model.state_dict(),
                    'model_args': model.config.__dict__
                }
                torch.save(ckpt, os.path.join(cfg.out_dir, 'ckpt.pt'))
                print(f"💾 Saved checkpoint to {cfg.out_dir}")
