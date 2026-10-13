# Biomass Brasil Pipeline — reproducible runtime
#
#   docker compose build            # builds this image as bbp:<version>
#   docker compose run --rm bbp info
#
# GDAL comes bundled inside the rasterio / pyogrio wheels: no system GDAL needed.
FROM python:3.12-slim

ARG GIT_COMMIT=""
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    BBP_RAW_ROOT=/data/raw \
    BBP_DATA_DIR=/app/data \
    BBP_GIT_COMMIT=${GIT_COMMIT}

WORKDIR /app

# dependencies first (cached layer), then the code
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install -e ".[dev]"

COPY config ./config
COPY tests ./tests

# run as an unprivileged user; data/ is a mounted volume
RUN useradd --create-home --uid 1000 bbp && mkdir -p /app/data && chown -R bbp /app
USER bbp

ENTRYPOINT ["bbp"]
CMD ["info"]
