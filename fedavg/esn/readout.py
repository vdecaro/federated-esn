import sys
from typing import Dict, Tuple, Iterator, Optional, List, Callable, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter

__all__ = ['Readout', 'fit_readout', 'fit_and_validate_readout']


class Readout(Module):
    """
    A linear readout
    Linear model with bias :math:`y = W x + b`, like Linear in Torch.
    """
    weight: Parameter  # (targets × features)
    bias: Parameter  # (targets)

    def __init__(self, num_features: int, num_targets: int):
        """
        New readout
        :param num_features: Number of input features
        :param num_targets: Number of output targets
        """
        super().__init__()
        self.weight = Parameter(Tensor(num_targets, num_features), requires_grad=False)
        self.bias = Parameter(Tensor(num_targets), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)

    def fit(self, data: Union[Iterator[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]],
            regularization: Union[Optional[float], List[float]] = None,
            validate: Optional[Callable[[Tuple[Tensor, Tensor]], float]] = None,
            verbose: bool = False):
        """
        Fit readout to data
        :param data: Dataset of (features, targets) tuples, or single pair
        :param regularization: Ridge regression lambda, or lambda if validation requested
        :param validate: Validation function, if regularization is to be selected
        :param verbose: Whether to print validation info (default false)
        """
        if not hasattr(data, '__next__'):
            data = iter([data])
        if callable(validate):
            self.weight.data, self.bias.data = fit_and_validate_readout(data, regularization, validate, verbose)
        else:
            self.weight.data, self.bias.data = fit_readout(data, regularization)

    @property
    def num_features(self) -> int:
        """
        Input features
        :return: Number of input features
        """
        return self.weight.shape[1]

    @property
    def num_targets(self) -> int:
        """
        Output targets
        :return: Number of output targets
        """
        return self.weight.shape[0]

    def __repr__(self):
        return f'Readout(features={self.num_features}, targets={self.num_targets})'


def compute_ridge_matrices(X: Tensor, Y: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Computes the matrices A and B for incremental ridge regression

    Args:
        X (Tensor): the input data (reservoir states) of shape [n_samples x hidden_size]
        Y (Tensor): the labels of shape [n_samples x label_size]

    Returns:
        Tuple[Tensor, Tensor]: the matrices A of shape [label_size x hidden_size] and B of shape [hidden_size x hidden_size]
    """
    A = Y.T @ X
    B = X @ X.T
    return A, B

def solve_ab_decomposition(A: Tensor, B: Tensor, l2: Optional[float] = None) -> Tensor:
    """
    Computes the result of the AB decomposition for solving the linear system

    Args:
        A (Tensor): YS^T
        B (Tensor): SS^T
        l2 (Optional[float], optional): The value of l2 regolarization. Defaults to None.

    Returns:
        Tensor: matrix W of shape [label_size x hidden_size]
    """
    B = B + torch.eye(B.shape[0]).to(B) * l2 if l2 else B
    return A @ B.inverse()

def fit_readout(X: Tensor, Y: Tensor, l2: Optional[float] = None) -> Tuple[Tensor, Tensor]:
    A, B = compute_ridge_matrices(X, Y)
    return solve_ab_decomposition(A, B, l2)


def fit_and_validate_readout(train_X: Tensor, 
                             train_Y: Tensor,
                             eval_X: Tensor,
                             eval_Y: Tensor,
                             l2_values: List[float],
                             score_fn: Callable[[Tensor, Tensor], Dict]) -> Tuple[Tensor, Tensor]:
    
    best_W, best_l2, best_eval_score = None, None, None
    A, B = compute_ridge_matrices(train_X, train_Y)
    for l2 in l2_values:
        W = solve_ab_decomposition(A, B, l2)
        Y_pred = F.linear(eval_X, W)
        score = score_fn(eval_Y, Y_pred)

        if best_W is None or score > best_eval_score:
            best_W, best_l2, best_eval_score = W, l2, score
    
    return best_W, best_l2, best_eval_score

def validate_readout(A: Tensor, 
                     B: Tensor,
                     eval_data,
                     l2_values: List[float],
                     score_fn: Callable[[Tensor, Tensor], Dict]) -> Tuple[Tensor, Tensor]:
    
    best_W, best_l2, best_eval_score = None, None, None
    for l2 in l2_values:
        W = solve_ab_decomposition(A, B, l2)
        acc, n_samples = 0, 0
        for x, y in eval_data:
            Y_pred = torch.argmax(F.linear(x, W), dim=-1).flatten()
            curr_acc = score_fn(torch.argmax(y, dim=-1).flatten(), Y_pred)
            curr_n_samples = Y_pred.size(0)
            acc += curr_acc*curr_n_samples
        acc = acc / n_samples

        if best_W is None or acc > best_eval_score:
            best_W, best_l2, best_eval_score = W, l2, acc
    
    return best_W, best_l2, best_eval_score