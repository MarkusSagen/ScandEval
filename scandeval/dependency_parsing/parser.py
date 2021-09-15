'''An implementation of the Biaffine Dependency Parser from [1].

[1] Timothy Dozat and Christopher D. Manning. 2017. Deep Biaffine Attention
    for Neural Dependency Parsing. In Proceedings of ICLR 2017.
'''

from transformers import AutoModelForTokenClassification
from typing import Optional, Tuple
import torch.nn as nn
import torch
from types import MethodType

from .modules import Biaffine, MLP
from .outputs import DependencyParsingOutput


class AutoModelForDependencyParsing(AutoModelForTokenClassification):

    @classmethod
    def from_pretrained(cls,
                        *args,
                        num_deps: int = 37,
                        head_dim: int = 100,
                        dep_dim: int = 100,
                        emb_dropout: float = 0.33,
                        head_dropout: float = 0.50,
                        dep_dropout: float = 0.50,
                        **kwargs):

        # Load model
        model = AutoModelForTokenClassification.from_pretrained(*args,
                                                                **kwargs)

        # Replace the dropout layer with a new one
        model.dropout = nn.Dropout(emb_dropout)

        # Replace the classifer head with the dependency parser
        parser = BiaffineDependencyParser(
            hidden_dim=model.config.hidden_size,
            head_dim=head_dim,
            dep_dim=dep_dim,
            num_deps=num_deps,
            head_dropout=head_dropout,
            dep_dropout=dep_dropout,
        )
        model.classifier = parser

        # Set the `forward` method
        model.old_forward = model.forward
        model.forward = MethodType(cls.forward, model)

        # Set `criterion` and `num_labels` attributes
        model.criterion = nn.CrossEntropyLoss()
        model.config.num_labels = num_deps

        # Return the model
        return model

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None,
                output_attentions: Optional[torch.Tensor] = None,
                output_hidden_states: Optional[torch.Tensor] = None):
        outputs = self.old_forward(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   inputs_embeds=inputs_embeds,
                                   labels=None,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states,
                                   return_dict=True)
        head_logits, dep_logits = outputs.logits

        # Extract the head labels and dep labels
        head_labels = labels[:, :, 0]
        dep_labels = labels[:, :, 1]

        loss = None
        if head_labels is not None and dep_labels is not None:

            # Define a mask for all the tokens that do not have the -100 label
            # and which is not a padding token
            mask = head_labels.ge(0)
            if attention_mask is not None:
                mask = mask & (attention_mask == 1)

            # Mask the predictions and labels accordingly
            active_head_logits = head_logits[mask]
            active_dep_logits = dep_logits[mask]
            active_head_labels = head_labels[mask]
            active_dep_labels = dep_labels[mask]

            # Get the dependency label logits for the gold arcs
            label_range = torch.arange(len(active_head_labels))
            active_dep_logits = active_dep_logits[label_range,
                                                  active_head_labels]

            # Compute the losses
            head_loss = self.criterion(active_head_logits, active_head_labels)
            dep_loss = self.criterion(active_dep_logits, active_dep_labels)
            loss = (head_loss + dep_loss) / 2

        return DependencyParsingOutput(
            loss=loss,
            head_logits=head_logits,
            dep_logits=dep_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


class BiaffineDependencyParser(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_deps: int,
                 head_dim: int,
                 dep_dim: int,
                 head_dropout: float,
                 dep_dropout: float):
        super().__init__()
        self.head_mlp_d = MLP(in_dim=hidden_dim,
                             out_dim=head_dim,
                             dropout=head_dropout)
        self.head_mlp_h = MLP(in_dim=hidden_dim,
                             out_dim=head_dim,
                             dropout=head_dropout)
        self.dep_mlp_d = MLP(in_dim=hidden_dim,
                             out_dim=dep_dim,
                             dropout=dep_dropout)
        self.dep_mlp_h = MLP(in_dim=hidden_dim,
                             out_dim=dep_dim,
                             dropout=dep_dropout)
        self.head_attn = Biaffine(in_dim=head_dim,
                                 bias_x=True,
                                 bias_y=False)
        self.dep_attn = Biaffine(in_dim=dep_dim,
                                 out_dim=num_deps,
                                 bias_x=True,
                                 bias_y=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        head_d = self.head_mlp_d(x)
        head_h = self.head_mlp_h(x)
        dep_d = self.dep_mlp_d(x)
        dep_h = self.dep_mlp_h(x)

        # [batch_size, seq_len, seq_len]
        s_head = self.head_attn(head_d, head_h)

        # [batch_size, seq_len, seq_len, n_deps]
        s_dep = self.dep_attn(dep_d, dep_h)

        return s_head, s_dep
