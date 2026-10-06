import torch
from autrainer.models import AbstractModel
from efficientnet_pytorch import EfficientNet as PTE

__version__ = "0.1.0"


class EfficientNet(AbstractModel):
    def __init__(
        self,
        output_dim: int,
        scaling_type: str = "efficientnet-b0",
        transfer: bool = False,
    ) -> None:
        super().__init__(output_dim)
        self.transfer = transfer
        self.scaling_type = scaling_type
        self.model = (PTE.from_pretrained if transfer else PTE.from_name)(scaling_type)
        self.linear = torch.nn.Linear(self.model._fc.in_features, output_dim)
        self.model._fc = torch.nn.Identity()

    def embeddings(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(self.embeddings(features))
