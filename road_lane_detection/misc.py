import time
import psutil
# https://docs.python.org/3/library/functools.html
from functools import wraps

# Decorator to measure runtime of a function
"""
Decorator: higher-order function that takes another function as an argument and extends its behavior without explicitly modifying it
"""
def calculate_runtime(func):
    @wraps(func)
    # this is the main "wrapper" function
    def wrapper(*args, **kwargs):
        # record the start time before calling the decorated function, or our desired function to analyze
        start_time = time.time()
        # function call
        result = func(*args, **kwargs)
        # record the end time right after the function call
        end_time = time.time()
        # you get what this does, right?
        runtime = end_time - start_time
        print(f"Function '{func.__name__}' took {runtime:.4f} seconds to execute")
        return result
    return wrapper

# Decorator to measure memory usage of a function
def calculate_memory_usage(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024.0 / 1024.0  # Memory in MB before function call
        result = func(*args, **kwargs)
        end_memory = process.memory_info().rss / 1024.0 / 1024.0  # Memory in MB after function call
        memory_usage = end_memory - start_memory
        print(f"Function '{func.__name__}' used approximately {memory_usage:.2f} MB of memory")
        return result
    return wrapper

# Decorator that logs function calls, including parameters and return values, can be useful for debugging
# and understanding the flow of your program
# Helps in tracking which functions are called and with what arguments
def log_function_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling function '{func.__name__}' with args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"Function '{func.__name__}' returned: {result}")
        return result
    return wrapper

# decorator that adds error handling and logging can centralize error management and provide consistent error handling
def handle_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            print(f"Error in function '{func.__name__}': {str(e)}")
            raise  # Re-raise the exception for propagation
        return result
    return wrapper

# Caches the results of expensive function calls can signigicantly improve performance by avoiding redundant computations
def cache(func):
    memo = {}
    @wraps(func)
    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            result = func(*args)
            memo[args] = result
            return result
    return wrapper

# Function that retries a function call a specified number of times on failure, which can
# be useful for handling transient errors
def retry(max_attempts, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {attempts+1} failed: {str(e)}")
                    attempts += 1
                    time.sleep(delay)
            raise Exception(f"Function '{func.__name__}' failed after {max_attempts} attempts")
        return wrapper
    return decorator