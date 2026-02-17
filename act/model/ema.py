"""
Exponential Moving Average (EMA) for ACT policy parameters.

Maintains a shadow copy of model weights updated as:
  theta_ema = decay * theta_ema + (1 - decay) * theta

The EMA model is used for inference only — gradients flow through the
original model during training, keeping the two copies independent.

Decay schedule (power-law warmup):
  decay(step) = 1 - (1 + step / inv_gamma)^(-power)
  - Early steps: decay ≈ 0  → EMA tracks weights quickly (fast adaptation)
  - Late steps:  decay → max_value  → strong smoothing (stable inference)
  power=2/3 and inv_gamma=1.0 are the defaults used in Diffusion Policy.

Buffer-aware update:
  Standard EMA implementations only update nn.Parameters and miss registered
  buffers (e.g. running_mean/var in BatchNorm, sinusoidal pos encoding tables).
  This implementation iterates buffers() alongside parameters() to keep the
  EMA model fully in sync with the live model.
"""

import copy

import torch


class EMAModel:
    """
    Exponential Moving Average of model parameters and buffers.

    Args:
        model: the live training model to track
        power: exponent for the power-law decay schedule (default 2/3)
        min_value: floor for the decay coefficient (default 0.0)
        max_value: ceiling for the decay coefficient (default 0.9999)
        update_after_step: number of steps before EMA starts updating (default 0)
        inv_gamma: inverse scale for the warmup (default 1.0)
    """

    def __init__(
        self,
        model,
        power=2 / 3,
        min_value=0.0,
        max_value=0.9999,
        update_after_step=0,
        inv_gamma=1.0,
    ):
        self.averaged_model = copy.deepcopy(model)
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma

        self.optimization_step = 0
        self.decay = 0.0

    def get_decay(self, optimization_step):
        """Compute decay coefficient at the given step."""
        step = max(0, optimization_step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        """
        Update EMA weights from the current model state.

        Iterates module-by-module to handle parameters and buffers separately:
        - Parameters with requires_grad=True: exponential moving average
        - Parameters with requires_grad=False: hard copy (frozen params like pos tables)
        - Buffers: hard copy (running stats, sinusoidal tables, etc.)
        """
        self.decay = self.get_decay(self.optimization_step)

        for module, ema_module in zip(new_model.modules(), self.averaged_model.modules()):
            # Parameters at this level only (recurse=False avoids double-counting)
            for param, ema_param in zip(
                module.parameters(recurse=False),
                ema_module.parameters(recurse=False),
            ):
                if not param.requires_grad:
                    ema_param.copy_(param.data)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(param.data, alpha=1 - self.decay)

            # Buffers: always hard-copied (pos encoding tables, BN running stats)
            for buf, ema_buf in zip(
                module.buffers(recurse=False),
                ema_module.buffers(recurse=False),
            ):
                ema_buf.copy_(buf.data)

        self.optimization_step += 1

    def copy_to(self, model):
        """
        Copy EMA parameters and buffers back into a model.

        Typically called at the end of training to produce the final
        inference-ready checkpoint, or when switching to EMA weights
        for evaluation.
        """
        for module, ema_module in zip(model.modules(), self.averaged_model.modules()):
            for param, ema_param in zip(
                module.parameters(recurse=False),
                ema_module.parameters(recurse=False),
            ):
                param.data.copy_(ema_param.data)
            for buf, ema_buf in zip(
                module.buffers(recurse=False),
                ema_module.buffers(recurse=False),
            ):
                buf.data.copy_(ema_buf.data)

    def state_dict(self):
        return {
            "averaged_model": self.averaged_model.state_dict(),
            "optimization_step": self.optimization_step,
            "decay": self.decay,
        }

    def load_state_dict(self, state_dict):
        self.averaged_model.load_state_dict(state_dict["averaged_model"])
        self.optimization_step = state_dict["optimization_step"]
        self.decay = state_dict["decay"]
