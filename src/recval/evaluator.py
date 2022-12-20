from dataclasses import dataclass, field


@dataclass
class RecEvaluator:
    metrics_list: list[str]
    cutoffs_list: list[int]
