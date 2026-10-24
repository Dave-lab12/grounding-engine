"""FastAPI application entrypoint for the Grounded backend."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.config import get_settings
from app.core.telemetry import configure_logging, get_logger, request_context_middleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load application settings during startup."""

    app.state.settings = get_settings()
    logger = get_logger("app.lifecycle")
    logger.info(
        "application_startup",
        service=app.state.settings.app_name,
        environment=app.state.settings.app_env,
        version=app.state.settings.app_version,
    )
    for warning in app.state.settings.startup_warnings():
        logger.warning("startup_warning", detail=warning)
    yield
    logger.info("application_shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )
    allowed_origins = [
        origin.strip()
        for origin in settings.cors_allowed_origins.split(",")
        if origin.strip()
    ]
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Trace-Id", "X-Run-Id", "X-Request-Id"],
        )
    app.middleware("http")(request_context_middleware)
    app.include_router(api_router)

    @app.get("/", tags=["meta"])
    async def root() -> dict[str, str]:
        """Return basic service metadata."""

        return {
            "service": settings.app_name,
            "environment": settings.app_env,
            "version": settings.app_version,
        }

    return app


app = create_app()
