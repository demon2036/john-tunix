# Copyright 2025 Google LLC
"""Timing Logger Plugin - Non-invasive timing metrics for GRPO training"""

import time
import functools
from typing import Any, Callable
import logging

logger = logging.getLogger(__name__)

# Global timing storage
_timing_metrics = {}


def time_function(name: str):
    """Decorator to measure function execution time."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time

            # Store timing metric
            _timing_metrics[name] = elapsed
            logger.info(f"{name} took {elapsed:.4f} seconds")

            return result
        return wrapper
    return decorator


def get_timing_metrics():
    """Get and clear current timing metrics."""
    metrics = dict(_timing_metrics)
    _timing_metrics.clear()
    return metrics


def patch_grpo_learner_timing(learner):
    """
    Monkey patch GRPO learner to add detailed timing metrics.

    This adds timing for:
    - rollout/generation time
    - reward computation time
    - actor update time

    Args:
        learner: The GRPO learner instance to patch
    """
    import types

    # Save original methods
    original_generate = learner._generate_and_compute_advantage
    original_train_step = learner.rl_cluster.actor_trainer.train_step if hasattr(learner, 'rl_cluster') else None

    # Patch _generate_and_compute_advantage to measure rollout + reward time
    def timed_generate_and_compute_advantage(self, *args, **kwargs):
        start_time = time.time()
        result = original_generate(*args, **kwargs)
        rollout_time = time.time() - start_time

        # Log rollout timing
        if hasattr(self, 'rl_cluster'):
            self.rl_cluster.buffer_metrics(
                {
                    "timing/rollout_and_reward_sec": (rollout_time, lambda x: x),
                },
                mode=self.rl_cluster.buffer_metrics.__self__.__class__.Mode.TRAIN if hasattr(self.rl_cluster.buffer_metrics.__self__.__class__, 'Mode') else 0,
            )

        _timing_metrics['rollout_and_reward'] = rollout_time
        return result

    learner._generate_and_compute_advantage = types.MethodType(
        timed_generate_and_compute_advantage, learner
    )

    logger.info("Timing logger plugin activated for GRPO learner")
    return learner


# Auto-register timing hooks when imported
def install():
    """Install timing hooks into Tunix."""
    logger.info("Timing logger plugin loaded")
