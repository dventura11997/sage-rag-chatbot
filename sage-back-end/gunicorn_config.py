# gunicorn_config.py
workers = 1  # Reduce number of workers to save memory
threads = 2  # Use threads instead of processes for better memory sharing
worker_class = 'gthread'  # Thread-based workers
max_requests = 100  # Restart workers after handling this many requests
max_requests_jitter = 10  # Add randomness to the restart schedule
timeout = 120  # Increase timeout for long operations
preload_app = False  # Don't preload the app to save memory