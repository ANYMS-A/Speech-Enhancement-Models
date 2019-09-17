from torch.utils.tensorboard import SummaryWriter
from resgan_model import ResGenerator
import torch

writer = SummaryWriter()

model = torch.nn.Linear(16384, 1)
dummy_input = torch.randn(1, 1, 16384)

with SummaryWriter(log_dir='logs') as w:
    w.add_graph(model, (dummy_input, ))

