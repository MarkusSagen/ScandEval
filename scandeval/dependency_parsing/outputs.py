'''Output class for the Biaffine Dependency Parser'''

from dataclasses import dataclass
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple
import torch


@dataclass
class DependencyParsingOutput(ModelOutput):
    '''Base class for outputs of dependency parsing models.

    Args:
        loss (PyTorch FloatTensor, optional) :
            Classification loss. Defaults to None.
        head_logits (PyTorch FloatTensor, optional):
            Classification scores for the heads, before softmax is applied.
            Defaults to None.
        dep_logits (PyTorch FloatTensor, optional):
            Classification scores for the dependency relations, before softmax
            is applied.  Defaults to None.
        hidden_states (Tuple of PyTorch FloatTensors, optional):
            One tensor for the output of the embeddings and one for the output
            of each layer, of shape [batch_size, sequence_length, hidden_size].
            Hidden-states of the model at the output of each layer plus the
            initial embedding outputs. Defaults to None.
        attentions (Tuple of PyTorch FloatTensors, optional):
            A tensor for each layer, of shape [batch_size, num_heads,
            seq_length, seq_length]. Attentions weights after the attention
            softmax, used to compute the weighted average in the self-attention
            heads. Defaults to None.
    '''

    loss: Optional[torch.FloatTensor] = None
    head_logits: Optional[torch.FloatTensor] = None
    dep_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
