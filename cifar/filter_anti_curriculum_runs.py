from hydra_filter_sweeper import AbstractFilter
from omegaconf import DictConfig


class FilterAntiCurriculumRuns(AbstractFilter):
    def filter(self, config: DictConfig, directory: str) -> bool:
        """Filter out runs using a random scoring function and anti-curriculum,
        as the order is random and has no sample difficulty meaning.

        Args:
            config: Run configuration.
            directory: Directory where the run configuration is stored.

        Returns:
            Whether to filter out the run configuration.
        """
        return (
            config.curriculum.type == "AntiCurriculum"
            and config.curriculum.scoring.id == "Random"
        )
