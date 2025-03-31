from stable_baselines3.common.callbacks import BaseCallback


class CustomCallbak(BaseCallback):
    def __inti__(self, verbose: int 0):
        super().__init(verbose)