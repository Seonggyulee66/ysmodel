# ---- schedulers.py ----
import math

class WarmupCosineWithRestarts:
    """
    - step_update(global_step)로 per-iteration 호출
    - warmup_steps 후 cosine schedule
    - 주기적 재시작(T_i): 1, 2, 4, 8... epoch 길이에 관계없이 'step' 기준.
    """
    def __init__(self, optimizer, base_lr=2e-4, min_lr=5e-5,
                 warmup_steps=5_000,      # 5 epoch 가정: epoch당 1k step이면 5*1000
                 first_cycle_steps=10_000, # 최초 10k step 후 restart
                 cycle_mult=2,            # 10k, 20k, 40k ...
                 last_step=-1):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult

        self.cycle = 0
        self.cycle_steps = first_cycle_steps
        self.cycle_step = 0
        self.last_step = last_step
        self._set_lr(min_lr if warmup_steps > 0 else base_lr)

    def _set_lr(self, lr):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def get_lr(self, global_step):
        # 1) Warmup
        if global_step < self.warmup_steps:
            # linear warmup: min_lr -> base_lr
            t = global_step / max(1, self.warmup_steps)
            return self.min_lr + (self.base_lr - self.min_lr) * t

        # 2) Cosine within current cycle
        t = self.cycle_step / max(1, self.cycle_steps)
        # cosine 0..1 -> base_lr..min_lr
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * t))

    def step_update(self, global_step):
        self.last_step = global_step

        # warmup 단계면 cycle 증가 없이 진행
        if global_step < self.warmup_steps:
            lr = self.get_lr(global_step)
            self._set_lr(lr)
            return lr

        # cycle 내 step 갱신
        self.cycle_step += 1
        # cycle 종료 -> 재시작
        if self.cycle_step >= self.cycle_steps:
            self.cycle += 1
            self.cycle_step = 0
            self.cycle_steps = int(self.cycle_steps * self.cycle_mult)

        lr = self.get_lr(global_step)
        self._set_lr(lr)
        return lr
