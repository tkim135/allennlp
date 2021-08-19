from overrides import overrides
from transformers import AutoModel
import torch
from torch import nn

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


@TokenEmbedder.register("gpt_pretrained")
class GptPretrainedEmbedder(TokenEmbedder):
    """
    Uses a pretrained model from ``pytorch-transformers`` as a ``TokenEmbedder``.
    """
    def __init__(self, model_name: str, num_classes: int) -> None:
        super().__init__()
        self.transformer_model = AutoModel.from_pretrained(model_name)
        # I'm not sure if this works for all models; open an issue on github if you find a case
        # where it doesn't work.
        self.output_dim = self.transformer_model.config.hidden_size
        self.score = nn.Linear(self.output_dim, num_classes, bias=False)

        self.init_weights(self.score)

        self.transformer_model.config.pad_token_id = self.transformer_model.config.eos_token_id

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.transformer_model.config.initializer_range)
            if isinstance(module, (nn.Linear)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @overrides
    def get_output_dim(self):
        return self.output_dim

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:  # type: ignore
        # pylint: disable=arguments-differ
        hidden_outputs = self.transformer_model(token_ids)[0]
        logits = self.score(hidden_outputs)
        batch_size, sequence_length = token_ids.shape[:2]
        assert (
            self.transformer_model.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        sequence_lengths = torch.ne(token_ids, self.transformer_model.config.pad_token_id).sum(-1) - 1
        pooled_logits = logits[range(batch_size), sequence_lengths]
        return pooled_logits
