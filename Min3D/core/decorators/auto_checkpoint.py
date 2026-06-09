# dependencies
import functools
from typing import Any, Callable

def auto_checkpoint(method: Callable) -> Any:
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs) -> Any:
        if not self.disable_auto_checkpoints:
            # write checkpoint before method execution
            self.create_checkpoint(slot="latest")
        return method(self, *args, **kwargs)

    return wrapper
