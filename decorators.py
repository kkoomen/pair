import time
import functools
from helpers import format_duration

def timer(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        duration = time.time() - start_time
        if hasattr(self, 'logger'):
            self.logger.debug(f"{func.__name__} took {format_duration(duration)}")
        else:
            print(f"{func.__name__} took {format_duration(duration)}")
        return result
    return wrapper
