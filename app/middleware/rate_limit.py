"""
app/middleware/rate_limit.py
IP-based sliding-window rate limiter.
Stores hit counts in-memory (swap dict for Redis in production).
Returns proper ERR_008 error code on breach.
"""
from __future__ import annotations
import time
import logging
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Paths that are never rate-limited
_EXEMPT = {"/", "/docs", "/redoc", "/openapi.json", "/health"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests   = max_requests
        self.window_seconds = window_seconds
        self._hits: dict    = defaultdict(list)   # {ip: [timestamp, ...]}

    # ── Helpers ─────────────────────────────────────────────────

    def _client_ip(self, request: Request) -> str:
        fwd = request.headers.get("X-Forwarded-For")
        if fwd:
            return fwd.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    # ── Middleware dispatch ──────────────────────────────────────

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _EXEMPT:
            return await call_next(request)

        ip  = self._client_ip(request)
        now = time.monotonic()
        win = now - self.window_seconds

        # Prune expired hits
        self._hits[ip] = [ts for ts in self._hits[ip] if ts > win]

        count = len(self._hits[ip])

        if count >= self.max_requests:
            logger.warning(f"Rate limit hit: {ip} on {request.url.path}")
            return JSONResponse(
                status_code=429,
                content={
                    "error_code": "ERR_008",
                    "message":    "Rate limit exceeded",
                    "detail":     (
                        f"Max {self.max_requests} requests per "
                        f"{self.window_seconds}s. Retry after {self.window_seconds}s."
                    ),
                },
                headers={"Retry-After": str(self.window_seconds)},
            )

        self._hits[ip].append(now)

        response = await call_next(request)

        # Inject rate-limit headers on every response
        remaining = max(0, self.max_requests - count - 1)
        response.headers["X-RateLimit-Limit"]     = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Window"]    = str(self.window_seconds)

        return response
