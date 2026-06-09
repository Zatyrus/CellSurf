# dependencies
import datetime
import functools
from typing import Any, Callable

def auto_log(method: Callable) -> Any:
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs) -> Any:
        # local! so can be same name
        if not self.disable_auto_log:
            # pedantic:
            assert list(self.log.keys())[-1] == max(self.log.keys()), KeyError(
                "dictionary not ordered enough"
            )  # make sure the latest is the greatest

            self.log[max(self.log.keys()) + 1] = {
                "date": datetime.datetime.now().strftime("%Y-%m-%dT_%H-%M-%S"),
                "method name": method.__name__,
                "args": args,
                "kwargs": kwargs,
            }
        return method(self, *args, **kwargs)

    return wrapper
