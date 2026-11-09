import logging
from datetime import datetime
from pathlib import Path

_configured = False


def get_logger(name="log"):
    global _configured
    if not _configured:
        log_dir = Path("log_files")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"log_{datetime.now().strftime('%Y-%m-%d')}.log"

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file),
            ],
        )
        _configured = True

    return logging.getLogger(name)
