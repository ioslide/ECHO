import math
from tqdm import tqdm
import torch
from torch import nn
from tqdm import trange

def clean_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int=100, is_enable_progress_bar: bool=False, eval_mode: bool=False, device: torch.device=None):
    if device is None:
        device = x.device
    acc = 0.0
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        if is_enable_progress_bar:
            iterator = trange(n_batches)
        else:
            iterator = range(n_batches)
        for counter in iterator:
            x_curr = x[counter * batch_size:(counter + 1) * batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) * batch_size].to(device)
            if eval_mode:
                output = model.forward(x_curr, y_curr, adapt=False)
            else:
                output = model(x_curr, y_curr)
            acc += (output.max(1)[1] == y_curr).float().sum()
    return acc.item() / x.shape[0]

def clean_accuracy_loader(model: nn.Module, loader, is_enable_progress_bar: bool=False, eval_mode: bool=False, device: torch.device=None):
    """Streaming accuracy/adaptation over a DataLoader.

    This avoids materializing all images for a corruption in host memory. It is
    env-gated from L-CS.py and preserves clean_accuracy's model-call semantics.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    acc = 0.0
    n = 0
    iterator = loader
    if is_enable_progress_bar:
        iterator = tqdm(loader)
    with torch.no_grad():
        for batch in iterator:
            if len(batch) >= 2:
                (x_curr, y_curr) = (batch[0], batch[1])
            else:
                raise ValueError('Expected loader batch to contain at least x and y')
            x_curr = x_curr.to(device)
            y_curr = y_curr.to(device)
            if eval_mode:
                output = model.forward(x_curr, y_curr, adapt=False)
            else:
                output = model(x_curr, y_curr)
            acc += (output.max(1)[1] == y_curr).float().sum().item()
            n += y_curr.numel()
    return acc / max(1, n)
