# docker/back/Dockerfile
FROM mnist-base:latest
WORKDIR /app
ENV PYTHONPATH=/app/src
COPY src ./src
WORKDIR /app/src/app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]