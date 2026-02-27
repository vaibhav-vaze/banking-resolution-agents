# 1. Start with a clean, official version of Python 3.11
FROM python:3.11-slim

# 2. Create a folder named /app inside the container and move into it
WORKDIR /app

# 3. Copy our requirements file from the Mac into the container
COPY requirements.txt .

# 4. Install the Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all our python files and folders into the container
COPY . .

# 6. Expose the port Gradio uses so we can access the UI from a web browser
EXPOSE 8501

# 7. Tell the container exactly what command to run when it wakes up
CMD ["python", "app.py"]