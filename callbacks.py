import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class SaveOnBestEpLenCallback(BaseCallback):
    """Save the model whenever the rolling mean episode length hits a new minimum.

    Uses `model.ep_info_buffer` (a deque of the last `stats_window_size` episodes).
    Waits until `min_episodes` are available so the rolling mean isn't noisy.
    """

    def __init__(self, save_path: str, min_episodes: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.min_episodes = int(min_episodes)
        self.best_mean_len = float("inf")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        buf = getattr(self.model, "ep_info_buffer", None)
        if buf is None or len(buf) < self.min_episodes:
            return
        mean_len = float(np.mean([ep["l"] for ep in buf]))
        if mean_len < self.best_mean_len:
            self.best_mean_len = mean_len
            os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
            self.model.save(self.save_path)
            if self.verbose:
                print(
                    f"[SaveOnBest] New best rolling ep_len={mean_len:.2f} "
                    f"at step {self.num_timesteps} -> {self.save_path}"
                )


class CheckpointAfterCallback(BaseCallback):
    """Save model every `freq` steps once `start` timesteps have been reached."""

    def __init__(self, save_path_template: str, start: int, freq: int, verbose=1):
        super().__init__(verbose)
        self.save_path_template = save_path_template
        self.start = int(start)
        self.freq = int(freq)
        self._last_saved_at = 0

    def _on_step(self) -> bool:
        t = self.num_timesteps
        if t < self.start:
            return True
        # Determine which checkpoint boundary we just crossed
        checkpoint = (t // self.freq) * self.freq
        if checkpoint > self._last_saved_at and checkpoint >= self.start:
            self._last_saved_at = checkpoint
            path = self.save_path_template.format(steps=checkpoint)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self.model.save(path)
            if self.verbose:
                print(f"[Checkpoint] Saved model at {t} timesteps -> {path}")
        return True


# print number of episodes finished in each rollout
class EpisodeCounterCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.completed_this_iter = 0

    def _on_rollout_start(self) -> None:
        self.completed_this_iter = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.completed_this_iter += 1
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("debug/episodes_finished_in_rollout", self.completed_this_iter)
