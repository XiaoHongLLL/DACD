import torch
from torchdiffeq import odeint
from typing import Callable, Optional, Sequence, Union


class ODESolver:
    pass

    def __init__(self, velocity_model: Callable):
        pass
        super().__init__()
        self.velocity_model = velocity_model

    def sample(
            self,
            x_init: torch.Tensor,
            time_grid: torch.Tensor,
            step_size: Optional[float] = None,
            method: str = "euler",
            atol: float = 1e-5,
            rtol: float = 1e-5,
            enable_grad: bool = False,
            **model_extras,
    ) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        pass
        time_grid = time_grid.to(x_init.device)

        def ode_func(t, x):


            return self.velocity_model(x=x, t=t, **model_extras)

        ode_opts = {"step_size": step_size} if step_size is not None else {}

        with torch.set_grad_enabled(enable_grad):
            sol = odeint(
                ode_func,
                x_init,
                time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )


        return sol[-1]
