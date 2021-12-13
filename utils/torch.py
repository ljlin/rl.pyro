import torch


class TorchHelper:

    def __init__(self, DEVICE=None):
        if DEVICE:
            self.device = DEVICE
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def f(self, x):
        return torch.tensor(x).float().to(self.device)

    def i(self, x):
        return torch.tensor(x).int().to(self.device)

    def l(self, x):
        return torch.tensor(x).long().to(self.device)

    def b(self, x):
        return torch.tensor(x).bool().to(self.device)


def init_weights(net):
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

class Clamp(torch.nn.Module):
    def __init__(self, max, min):
        super().__init__()
        self.max = max
        self.min = min

    def forward(self, input):
        return torch.clamp(input, max=self.max, min = self.min)

class LogSoftmax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits):
        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = logits - torch.logsumexp(logits, dim=-1).view(-1,1)
        return probs, log_probs
