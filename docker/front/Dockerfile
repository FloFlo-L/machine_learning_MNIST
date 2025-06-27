# docker/front/Dockerfile
FROM mnist-base:latest
WORKDIR /app
COPY src/app/app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]