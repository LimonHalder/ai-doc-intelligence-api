from slowapi import Limiter
from slowapi.util import get_remote_address

# Keyed by client IP. In production behind a proxy, configure
# slowapi to read X-Forwarded-For instead.
limiter = Limiter(key_func=get_remote_address)
