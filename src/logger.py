import logging
import os
from datetime import datetime

# Format datetime safely for filename (no slashes or colons)
LOG_FILE = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log')

# Create logs directory path
logs_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Full log file path
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

if __name__ == "__main__":
    logging.info("Logging has started")
