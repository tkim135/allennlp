from overrides import overrides
from transformers import AutoModel
import torch

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


@TokenEmbedder.register("gpt_pretrained")
class GptPretrainedEmbedder(TokenEmbedder):
    """
    Uses a pretrained model from ``pytorch-transformers`` as a ``TokenEmbedder``.
    """
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.transformer_model = AutoModel.from_pretrained(model_name)
        # I'm not sure if this works for all models; open an issue on github if you find a case
        # where it doesn't work.
        self.output_dim = self.transformer_model.config.hidden_size
        self.transformer_model.config.pad_token_id = self.transformer_model.config.eos_token_id

    @overrides
    def get_output_dim(self):
        return self.output_dim

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:  # type: ignore
        # pylint: disable=arguments-differ
        hidden_outputs = self.transformer_model(token_ids)[0]
        batch_size, sequence_length = token_ids.shape[:2]
        assert (
            self.transformer_model.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        sequence_lengths = torch.ne(token_ids, self.transformer_model.config.pad_token_id).sum(-1) - 1
        return hidden_outputs[range(batch_size), sequence_lengths]
