"""Small CPU checks of RGB keys, memory limits, optimizer state and reset."""
from copy import deepcopy
import unittest
import torch
from torch import nn
from conf import cfg as defaults
from methods import Tent
from methods.ECHO import wrap_adapter
from methods.ECHO.basic_keys import RGBHistogramKey


class EchoTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(9)
        self.cfg = defaults.clone()
        self.cfg.ADAPTER.ECHO.ENABLED = True
        self.cfg.ADAPTER.ECHO.FEATURES = 16
        self.cfg.ADAPTER.ECHO.CALIBRATION = 3
        self.model = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(3, 4, 1),
                                   nn.BatchNorm2d(4), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 3))

    def test_rgb_histogram_mass_and_endpoints(self):
        x = torch.zeros(2, 3, 4, 4)
        x[:, 1] = 1
        key = RGBHistogramKey()(x).view(3, 8)
        torch.testing.assert_close(key.sum(1), torch.ones(3))
        self.assertEqual(key[0, 0], 1)
        self.assertEqual(key[1, -1], 1)

    def test_constant_kernel_matches_continual_tent(self):
        self.cfg.ADAPTER.ECHO.SHARE = 1.0
        self.cfg.ADAPTER.ECHO.GLOBAL_DECAY = 1.0
        self.cfg.ADAPTER.ECHO.LOCAL_DECAY = 1.0
        plain = Tent.setup(deepcopy(self.model), self.cfg)
        echo = wrap_adapter(Tent.setup(deepcopy(self.model), self.cfg), self.cfg)
        for _ in range(8):
            x = torch.rand(4, 3, 8, 8)
            torch.testing.assert_close(echo(x), plain(x), atol=2e-6, rtol=2e-5)
            for a, b in zip(echo.model.parameters(), plain.model.parameters()):
                torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)

    def test_reset_and_fixed_memory(self):
        echo = wrap_adapter(Tent.setup(deepcopy(self.model), self.cfg), self.cfg)
        shapes = []
        for _ in range(10):
            echo(torch.rand(4, 3, 8, 8))
            if echo.memory.A is not None:
                shapes.append(tuple(echo.memory.A.shape))
        self.assertEqual(len(set(shapes)), 1)
        self.assertTrue(echo.diagnostics()['moments_in_memory'])
        echo.reset()
        self.assertIsNone(echo.memory.A)
        for name, value in self.model.state_dict().items():
            if name in echo.adapter.model_state:
                torch.testing.assert_close(echo.model.state_dict()[name], echo.adapter.model_state[name])


if __name__ == '__main__':
    unittest.main()
