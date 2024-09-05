from tokenizers import Tokenizer
from torch import nn


class W2V_CBOW(nn.Module):
    """Implements the Continuous Bag of Words Word2Vec model."""

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        embedding_dim: int = 512,
        neighborhood_size: int = 2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.tokenizer = tokenizer
        self.neighborhood_size = neighborhood_size
        self.chunk_size = self.neighborhood_size * 2 + 1
        self.vocab_size = tokenizer.get_vocab_size()
        self.projection_dim = embedding_dim
        self.projection_layer = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=embedding_dim, max_norm=1
        )
        self.linear_layer = nn.Linear(
            in_features=embedding_dim, out_features=self.vocab_size
        )

    def forward(self, x):
        x = self.projection_layer(x)
        x = x.mean(dim=1)
        x = self.linear_layer(x)
        return x

    # def on_train_start(self):
    #     self.logger.log_hyperparams(
    #         self.hparams,
    #         {
    #             "hp/embed_dim": self.embedding_dim,
    #             "hp/neighborhood_size": self.neighborhood_size,
    #         },
    #     )

    # def training_step(self, batch, batch_idx):
    #     y_hat = self(batch["features"])
    #     loss = F.cross_entropy(
    #         y_hat,
    #         batch["labels"],
    #     )
    #     self.log("train_loss", loss)
    #     return loss
