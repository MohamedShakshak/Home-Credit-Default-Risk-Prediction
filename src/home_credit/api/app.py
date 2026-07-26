"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from home_credit.api.deps import get_settings, load_model
from home_credit.api.middleware import RequestIDMiddleware
from home_credit.api.routes.explain import router as explain_router
from home_credit.api.routes.predict import router as predict_router
from home_credit.api.routes.predict_batch import router as predict_batch_router
from home_credit.api.schemas import HealthResponse

try:
    from prometheus_fastapi_instrumentator import Instrumentator  # noqa: F401, F811
except ImportError:
    pass

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model on startup, clean up on shutdown."""
    try:
        load_model()
        app.state.model_loaded = True
    except Exception as exc:
        app.state.model_loaded = False
        structlog.get_logger().error("model_load_failed", error=str(exc))
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Home Credit Default Risk API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIDMiddleware)

    app.include_router(predict_router)
    app.include_router(predict_batch_router)
    app.include_router(explain_router)

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        settings = get_settings()
        return HealthResponse(
            status="ok" if getattr(app.state, "model_loaded", False) else "degraded",
            model_loaded=getattr(app.state, "model_loaded", False),
            model_stage=settings.model_stage,
        )

    with suppress(Exception):
        Instrumentator().instrument(app).expose(app)

    return app


app = create_app()
