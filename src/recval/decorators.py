import logging
import time
from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")


def timeit(func: Callable[P, T]) -> Callable[P, T]:
    """
    Measures execution time of a function.
    """

    @wraps(func)
    def timeit_wrapper(*args: Any, **kwargs: Any) -> T:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logging.debug("Function %s Took %.2f seconds", func.__name__, total_time)
        return result

    return timeit_wrapper
