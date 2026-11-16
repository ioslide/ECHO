"""CPU example: a base entropy step wrapped by ECHO, without downloads."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from conf import cfg as defaults
from methods import Tent
from methods.ECHO import wrap_adapter


def main():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    cfg = defaults.clone()
    cfg.ADAPTER.ECHO.ENABLED = True
    cfg.ADAPTER.ECHO.FEATURES = 128
    cfg.ADAPTER.ECHO.CALIBRATION = 4
    cfg.CORRUPTION.NUM_CLASS = 3
    source = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(3, 8, 3, padding=1),
                           nn.BatchNorm2d(8), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 3))
    adapter = wrap_adapter(Tent.setup(source, cfg), cfg)
    for step in range(12):
        batch = torch.rand(8, 3, 16, 16) * (0.3 if step % 4 < 2 else 1.0)
        output = adapter(batch)
        assert output.shape == (8, 3) and torch.isfinite(output).all()
    stats = adapter.diagnostics()
    print('ECHO demo complete: batches=%d, key_dim=%d, memory_shape=%s' %
          (stats['batches'], stats['key_dim'], stats['memory_shape']))


if __name__ == '__main__':
    main()
