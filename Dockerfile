FROM python:3.10-slim

# Keep Python from generating .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install the package and all dependencies via pyproject.toml
COPY scTenifoldXct/ /app/scTenifoldXct/
COPY pyproject.toml README.md /app/

RUN pip install --no-cache-dir .

# Copy example data and tutorials
COPY tutorials/ /app/tutorials/
COPY data/ /app/data/

# Create a non-root user
RUN adduser --uid 5678 --disabled-password --gecos "" appuser \
    && chown -R appuser /app
USER appuser

CMD ["sh"]
