"""CUDA values/gradients match the CPU reference; sampling crosses to NumPy."""

import unittest

import torch

from gridworld.gradients import differentiate_policy
from gridworld.model import Gridworld, State
from gridworld.policy import TabularSoftmaxPolicy
from gridworld.sampling import sample_episode
from gridworld.values import evaluate_policy


class DeviceSelectionTests(unittest.TestCase):
    def test_default_prefers_available_cuda_and_cpu_override_is_respected(self):
        state = State((0, 0), 1)
        automatic = TabularSoftmaxPolicy({state: [0, 1, 2, 3]})
        expected = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.assertEqual(automatic.logits[state].device.type, expected)
        cpu = TabularSoftmaxPolicy({state: [0, 1, 2, 3]}, device='cpu')
        self.assertEqual(cpu.probabilities(state).device.type, 'cpu')


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA device/driver unavailable')
class CudaTests(unittest.TestCase):
    def test_values_and_autograd_stay_on_cuda_and_match_cpu(self):
        world = Gridworld(layout=("SHG",), horizon=2, slip=0.3)
        logits = {s: [0, 1, 2, 3] for s in world.states()}
        cpu = differentiate_policy(world, TabularSoftmaxPolicy(logits, device='cpu'), gamma=0.7)
        policy = TabularSoftmaxPolicy(logits, device='cuda')
        gpu = differentiate_policy(world, policy, gamma=0.7)
        self.assertEqual(gpu.values.start_return.device.type, 'cuda')
        torch.testing.assert_close(gpu.values.start_return.cpu(), cpu.values.start_return,
                                   rtol=1e-12, atol=1e-12)
        for name in ('v', 'q', 'advantage'):
            for state, actual in getattr(gpu.values, name).items():
                self.assertEqual(actual.device.type, 'cuda')
                torch.testing.assert_close(actual.cpu(), getattr(cpu.values, name)[state],
                                           rtol=1e-12, atol=1e-12)
        for state, actual in gpu.gradient.items():
            self.assertEqual(actual.device.type, 'cuda')
            torch.testing.assert_close(actual.cpu(), cpu.gradient[state], rtol=1e-12, atol=1e-12)

        # Independently perturb a later-time CUDA parameter and reevaluate J.
        state, epsilon = State((0, 1), 1), 1e-5
        returns = []
        with torch.no_grad():
            for sign in (1, -1):
                rows = {s: row.detach().clone() for s, row in policy.logits.items()}
                rows[state][1] += sign * epsilon
                perturbed = TabularSoftmaxPolicy(rows, device='cuda')
                returns.append(evaluate_policy(world, perturbed, gamma=0.7).start_return.item())
        numerical = (returns[0] - returns[1]) / (2 * epsilon)
        self.assertAlmostEqual(gpu.gradient[state][1].item(), numerical, delta=1e-8)

    def test_cuda_policy_preserves_seeded_numpy_sampling(self):
        world = Gridworld(horizon=4)
        logits = {s: [0, 1, 2, 3] for s in world.states() if not world.is_terminal(s)}
        cpu = sample_episode(world, TabularSoftmaxPolicy(logits, device='cpu'), seed=7,
                             reward_noise_std=2)
        gpu = sample_episode(world, TabularSoftmaxPolicy(logits, device='cuda'), seed=7,
                             reward_noise_std=2)
        self.assertEqual(gpu, cpu)


if __name__ == '__main__':
    unittest.main()
