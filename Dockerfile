FROM python:3.13-slim

# working directory inside the container
WORKDIR /app

# copy just the requirements
COPY requirements.txt .

# install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the application code
COPY api/ ./api/
COPY models/ ./models/

# expose the port FastAPI runs on
EXPOSE 8000


# The command to run the app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]