"""Tests for ProcurementEnv — Gymnasium environment."""

import gymnasium
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from procurement_gym.env import ProcurementEnv


class TestProcurementEnvBasic:
    def test_create(self) -> None:
        env = ProcurementEnv()
        assert env.observation_space is not None
        assert env.action_space is not None

    def test_reset_returns_obs_info(self) -> None:
        env = ProcurementEnv()
        obs, info = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        assert "current_item" in obs
        assert "lot_summaries" in obs
        assert "global_context" in obs
        assert "action_mask" in obs

    def test_obs_in_space(self) -> None:
        env = ProcurementEnv()
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)

    def test_step_returns_five_tuple(self) -> None:
        env = ProcurementEnv()
        obs, _ = env.reset(seed=42)
        # Find a valid action from the mask
        mask = obs["action_mask"]
        valid_actions = np.where(mask == 1)[0]
        assert len(valid_actions) > 0
        obs2, reward, terminated, truncated, info = env.step(int(valid_actions[0]))
        assert isinstance(obs2, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_episode_terminates(self) -> None:
        """A full episode should terminate after N steps."""
        env = ProcurementEnv()
        obs, _ = env.reset(seed=42)
        done = False
        steps = 0
        while not done:
            mask = obs["action_mask"]
            valid = np.where(mask == 1)[0]
            obs, reward, terminated, truncated, _ = env.step(int(valid[0]))
            done = terminated or truncated
            steps += 1
            if steps > 600:
                pytest.fail("Episode did not terminate within expected steps")
        assert terminated
        assert not truncated
        assert reward != 0.0  # terminal reward should be nonzero

    def test_action_mask_first_step(self) -> None:
        """First step: only action 0 (new lot) should be valid."""
        env = ProcurementEnv()
        obs, _ = env.reset(seed=42)
        mask = obs["action_mask"]
        assert mask[0] == 1  # can open lot 0
        assert sum(mask) == 1  # only one valid action on first step

    def test_action_mask_second_step(self) -> None:
        """After first assignment, can assign to lot 0 or open lot 1."""
        env = ProcurementEnv()
        obs, _ = env.reset(seed=42)
        obs, _, _, _, _ = env.step(0)  # assign item 0 to lot 0
        mask = obs["action_mask"]
        assert mask[0] == 1  # existing lot 0
        assert mask[1] == 1  # new lot 1
        assert sum(mask) == 2

    def test_invalid_action_handled(self) -> None:
        """Invalid action (out of mask) should not crash — needed for check_env."""
        env = ProcurementEnv()
        obs, _ = env.reset(seed=42)
        # Action 5 is invalid on first step (only action 0 is valid)
        obs2, reward, terminated, truncated, info = env.step(5)
        # Should not crash; env handles gracefully (assigns to lot 0 instead)
        assert env.observation_space.contains(obs2)

    def test_seeded_reproducibility(self) -> None:
        env = ProcurementEnv()
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1["current_item"], obs2["current_item"])
        np.testing.assert_array_equal(obs1["lot_summaries"], obs2["lot_summaries"])

    def test_render_ansi(self) -> None:
        env = ProcurementEnv(render_mode="ansi")
        env.reset(seed=42)
        output = env.render()
        assert isinstance(output, str)
        assert len(output) > 0

    def test_close_safe(self) -> None:
        env = ProcurementEnv()
        env.reset(seed=42)
        env.close()
        env.close()  # double close should not crash


class TestCheckEnv:
    def test_gymnasium_check_env(self) -> None:
        """The critical exit criterion: check_env must pass."""
        env = ProcurementEnv()
        check_env(env)
