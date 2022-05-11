from typing import Optional, Callable, Union
from numpy import require

import torch
import torch.nn.functional as F
from torch import Tensor, Size
from torch.nn import Module, Parameter

from . import initializers

__all__ = ['Reservoir']


class Reservoir(Module):
    """
    A Reservoir of for Echo State Networks
    
    Args:
        input_size: the number of expected features in the input `x`
        hidden_size: the number of features in the hidden state `h`
        activation: name of the activation function from `torch` (e.g. `torch.tanh`)
        leakage: the value of the leaking parameter `alpha`
        input_scaling: the value for the desired scaling of the input (must be `<= 1`)
        rho: the desired spectral radius of the recurrent matrix (must be `< 1`)
        bias: if ``False``, the layer does not use bias weights `b`
        mode: execution mode of the reservoir (vanilla or intrinsic plasticity)
        kernel_initializer: the kind of initialization of the input transformation. Default: `'uniform'`
        recurrent_initializer: the kind of initialization of the recurrent matrix. Default: `'normal'`
    """

    def __init__(self,
                 input_size: int, 
                 hidden_size: int,
                 activation: str,
                 leakage: float = 1.,
                 input_scaling: float = 0.9,
                 rho: float = 0.99,
                 bias: bool = False,
                 kernel_initializer: Union[str, Callable[[Size], Tensor]] = 'uniform',
                 recurrent_initializer: Union[str, Callable[[Size], Tensor]] = 'normal',
                 mode: str = 'vanilla',
                 mu: Optional[float] = None,
                 sigma: Optional[float] = None) -> None:
        super().__init__()
        assert mode in ['vanilla', 'intrinsic_plasticity']
        if mode == 'intrinsic_plasticity':
            assert mu is not None and sigma is not None
        assert rho < 1 and input_scaling <= 1

        self.input_scaling = Parameter(torch.tensor(input_scaling), requires_grad=False)
        self.rho = Parameter(torch.tensor(rho), requires_grad=False)

        self.W_in = Parameter(
            init_params(kernel_initializer, scale=input_scaling)([hidden_size, input_size]), 
            requires_grad=False
        ) 
        self.W_hat = Parameter(
            init_params(recurrent_initializer, rho=rho)([hidden_size, hidden_size]),
            requires_grad=False
        )
        self.b = Parameter(
            init_params('uniform', scale=input_scaling)(hidden_size), 
            requires_grad=False
        ) if bias else None
        self.f = getattr(torch, activation)

        self.alpha = Parameter(torch.tensor(leakage), requires_grad=False)

        self.mode = mode
        if mode == 'intrinsic_plasticity':
            self.ip_a = Parameter(
                init_params('ones', scale=1)(hidden_size),
                requires_grad=True
            )
            self.ip_b = Parameter(
                init_params('zeros', scale=0)(hidden_size),
                requires_grad=True
            )
            self.mu = Parameter(torch.tensor(mu), requires_grad=False)
            self.v = Parameter(torch.tensor(sigma**2), requires_grad=False)


    def forward(self, input: Tensor, initial_state: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> Tensor:
        if self.mode == 'vanilla':
            reservoir_f = self._vanilla_state_comp
        if self.mode == 'intrinsic_plasticity':
            reservoir_f = self._ip_state_comp

        if initial_state is None:
            initial_state = torch.zeros(self.hidden_size).to(self.W_hat)
        
        embeddings = torch.stack([state for state in reservoir_f(input.to(self.W_hat), initial_state, mask)], dim=0)
        
        return embeddings


    def _vanilla_state_comp(self, input: Tensor, initial_state: Tensor, mask: Optional[Tensor] = None):
        timesteps = input.shape[0]
        state = initial_state
        for t in range(timesteps):
            h_t = torch.tanh(F.linear(input[t], self.W_in, self.b) + F.linear(state, self.W_hat))
            state = (1 - self.alpha) * state + self.alpha * h_t
            yield state if mask is None else mask * state


    def _ip_state_comp(self, input: Tensor, initial_state: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        timesteps = input.shape[0]
        in_signal = []
        h = []
        state = initial_state
        for t in range(timesteps):
            in_signal_t = F.linear(input[t], self.W_in, self.b) + F.linear(state, self.W_hat)
            h_t = torch.tanh(in_signal_t * self.ip_a + self.ip_b)
            state = (1 - self.alpha) * state + self.alpha * h_t
            yield state if mask is None else mask * state
            in_signal.append(in_signal_t)
            h.append(h_t)

        if self.training:
            in_signal, h = torch.stack(in_signal, dim=0), torch.stack(h, dim=0)
            ip_b_grad = -self.mu/self.v + (h/self.v)*(2*self.v + 1 - h**2 + self.mu*h)
            ip_a_grad = -1/self.ip_a + ip_b_grad*in_signal
            if mask is not None:
                ip_b_grad = mask * ip_b_grad
                ip_a_grad = mask * ip_a_grad
            if ip_b_grad.dim() > 1:
                ip_b_grad = ip_b_grad.sum(0)
                ip_a_grad = ip_a_grad.sum(0)
            if ip_b_grad.dim() > 1:
                ip_b_grad = ip_b_grad.mean(0)
                ip_a_grad = ip_a_grad.mean(0)
            
            self.ip_b.grad = -ip_b_grad
            self.ip_a.grad = -ip_a_grad
            


    @property
    def input_size(self) -> int:
        """Input dimension"""
        return self.W_in.shape[1]


    @property
    def hidden_size(self) -> int:
        """Reservoir state dimension"""
        return self.W_hat.shape[1]


def init_params(name: str, **options) -> Callable[[Size], Tensor]:
    """
    Gets a random weight initializer
    :param name: Name of the random matrix generator in `esn.initializers`
    :param options: Random matrix generator options
    :return: A random weight generator function
    """
    init = getattr(initializers, name)
    return lambda size: init(size, **options)