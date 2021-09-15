'''Various modules needed to implement the Biaffine Dependency Parser.

These modules have been adapted from the yzhangcs/parser package [1].

[1]: https://github.com/yzhangcs/parser
'''

import torch
import torch.nn as nn


class Biaffine(nn.Module):
    '''Biaffine module.

    Args:
        TODO
    '''
    def __init__(self,
                 in_dim: int,
                 out_dim: int = 1,
                 scale: float = 1.,
                 bias_x: bool = True,
                 bias_y: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.scale = scale
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(out_dim,
                                                in_dim + bias_x,
                                                in_dim + bias_y))
        self.reset_parameters()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)

        # [batch_size, seq_len, seq_len, out_dim]
        s = torch.einsum('bxi,oij,byj->bxyo', x, self.weight, y) / self.in_dim
        s = s ** self.scale

        # remove last dimension if out_dim == 1
        s = s.squeeze(-1)

        return s

    def __repr__(self) -> str:
        s = f"in_dim={self.in_dim}"
        if self.out_dim > 1:
            s += f", out_dim={self.out_dim}"
        if self.scale != 1:
            s += f", scale={self.scale}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)


class MLP(nn.Module):
    '''A Multi-Layered Perceptron.

    Args:
        in_dim (int):
            The input dimension.
        out_dim (int):
            The output dimension.
        dropout (float, optional):
            The percentage of units to dropout. Note that the dropout used here
            is shared dropout, meaning that if a unit is being dropped out then
            all units along the xx'th dimension will also be dropped out.
            Defaults to 0.0.
        activation (bool, optional):
            Whether a GELU activation should be applied
    '''
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 dropout: float = 0.0,
                 activation: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.GELU() if activation else nn.Identity()
        self.dropout = SharedDropout(dropout)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

    def __repr__(self):
        s = f'in_dim={self.in_dim}, out_dim={self.out_dim}'
        if self.dropout.p > 0:
            s += f', dropout={self.dropout.p}'
        return f'{self.__class__.__name__}({s})'

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)


class SharedDropout(nn.Module):
    '''Shared dropout module.

    SharedDropout differs from the vanilla dropout strategy in that the dropout
    mask is shared across one dimension.

    Args:
        p (float, optional):
            The percentage of units to dropout. Defaults to 0.5.
        batch_first (bool, optional):
            Whether the input shape is [batch_size, seq_len, *], rather than
            [seq_len, batch_size, *]. Defaults to True.

    Examples:
        >>> x = torch.ones(1, 3, 5)
        >>> x
        tensor([[[1., 1., 1., 1., 1.],
                 [1., 1., 1., 1., 1.],
                 [1., 1., 1., 1., 1.]]])
        >>> nn.Dropout()(x)
        tensor([[[0., 1., 1., 0., 0.],
                 [1., 1., 0., 1., 1.],
                 [1., 1., 1., 1., 0.]]])
        >>> SharedDropout()(x)
        tensor([[[1., 0., 1., 0., 1.],
                 [1., 0., 1., 0., 1.],
                 [1., 0., 1., 0., 1.]]])
    '''
    def __init__(self, p: float = 0.5, batch_first: bool = True):
        super().__init__()
        self.p = p
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            if self.batch_first:
                mask = self.get_mask(x[:, 0], self.p).unsqueeze(1)
            else:
                mask = self.get_mask(x[0], self.p)
            x = x * mask
        return x

    @staticmethod
    def get_mask(x: torch.Tensor, p: float) -> torch.Tensor:
        return x.new_empty(x.shape).bernoulli_(1 - p) / (1 - p)

    def __repr__(self) -> str:
        s = f'p={self.p}'
        if self.batch_first:
            s += f', batch_first={self.batch_first}'
        return f'{self.__class__.__name__}({s})'
