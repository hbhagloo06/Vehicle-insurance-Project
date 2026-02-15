import logging
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler


PROJECT_ROOT=Path(__file__).resolve().parents[2]
LOG_DIR='logs'
MAX_LOG_SIZE=5*1024*1024  #5MB
BACKUP_COUNT=3

def configure_logging(level:int=logging.INFO):
    root_logger=logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    if root_logger.handlers:
        return
    log_dir_path=PROJECT_ROOT / LOG_DIR
    log_dir_path.mkdir(parents=True,exist_ok=True)

    log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

    log_file_path=log_dir_path / log_file
    formatter = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")

    file_handler=RotatingFileHandler(filename=log_file_path,maxBytes=MAX_LOG_SIZE,backupCount=BACKUP_COUNT)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler=logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    #reduce noisy third-party logs
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)





