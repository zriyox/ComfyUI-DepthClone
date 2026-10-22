import unittest

import torch

from nodes import DepthCloneDepth, motion_depth


class MotionDepthTest(unittest.TestCase):
    def test_detail_mode_keeps_depth_unchanged(self):
        depth = torch.rand(2, 96, 72, 3)

        result = motion_depth(depth, "DETAIL")

        self.assertIs(result, depth)

    def test_motion_mode_keeps_shape_and_reduces_high_frequency_detail(self):
        base = torch.linspace(0, 1, 72).view(1, 1, 72, 1).expand(1, 96, 72, 3)
        checker = ((torch.arange(96).view(96, 1) + torch.arange(72)) % 2).float()
        detailed = (base + checker.view(1, 96, 72, 1) * 0.2).clamp(0, 1)

        result = motion_depth(detailed, "MOTION")

        self.assertEqual(result.shape, detailed.shape)
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)
        source_variation = (detailed[:, 1:] - detailed[:, :-1]).abs().mean()
        result_variation = (result[:, 1:] - result[:, :-1]).abs().mean()
        self.assertLess(float(result_variation), float(source_variation) * 0.35)

    def test_motion_mode_uses_sixteen_depth_levels_before_resize(self):
        depth = torch.linspace(0, 1, 120 * 90).view(1, 120, 90, 1).repeat(1, 1, 1, 3)

        result = motion_depth(depth, "MOTION")

        center_values = result[0, ::8, ::8, 0]
        quantized = torch.round(center_values * 15) / 15
        self.assertLess(float((center_values - quantized).abs().mean()), 0.02)

    def test_motion_mode_does_not_create_dark_frame_borders(self):
        depth = torch.full((1, 960, 720, 3), 0.6)

        result = motion_depth(depth, "MOTION")

        self.assertLess(float((result - result[:, 480:481, 360:361]).abs().max()), 1e-6)

    def test_node_defaults_to_detail_for_existing_workflows(self):
        mode = DepthCloneDepth.INPUT_TYPES()["optional"]["mode"]

        self.assertEqual(mode[0], ["DETAIL", "MOTION"])
        self.assertEqual(mode[1]["default"], "DETAIL")


if __name__ == "__main__":
    unittest.main()
