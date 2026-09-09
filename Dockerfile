FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .
ENTRYPOINT ["battery-schedule"]
CMD ["run", "--config", "configs/default.yaml"]
