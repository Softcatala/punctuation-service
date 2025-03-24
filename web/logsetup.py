#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
import os
import logging
import logging.handlers
import time
from filelock import FileLock


class LogSetup:

    def __init__(self):
        self.LOGDIR = os.environ.get("LOGDIR", ".")
        log_id = self._get_log_id()
        self.logfile = os.path.join(self.LOGDIR, f"punctuation-service_{log_id}.log")

    # To be removed when we have: https://github.com/encode/uvicorn/pull/2529/files
    def _get_log_id(self):
        FILENAME = "id.txt"

        lock_file = os.path.join(self.LOGDIR, f"{FILENAME}.lock")
        lock = FileLock(lock_file)
        id_file = os.path.join(self.LOGDIR, f"{FILENAME}")

        while True:
            try:
                with lock:
                    if not os.path.exists(id_file):
                        counter = 0
                    else:
                        with open(id_file, "r") as f:
                            try:
                                counter = int(f.read().strip())
                            except ValueError:
                                counter = 0

                    counter += 1
                    with open(id_file, "w") as f:
                        f.write(str(counter))

                    return counter
            except Exception as e:
                print(f"Error: {e}, retrying...")
            time.sleep(1)

    def init_logging(self):
        LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
        logger = logging.getLogger()
        # Rotate every 15 days, keep last 3 logs
        hdlr = logging.handlers.TimedRotatingFileHandler(
            self.logfile, when="D", interval=15, backupCount=3
        )
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        logger.setLevel(LOGLEVEL)

        console = logging.StreamHandler()
        console.setLevel(LOGLEVEL)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console.setFormatter(formatter)
        logger.addHandler(console)

    def get_logfiles(self):

        return [
            os.path.join(self.LOGDIR, f)
            for f in os.listdir(self.LOGDIR)
            if f.startswith("punctuation-service_") and f.endswith(".log")
        ]
