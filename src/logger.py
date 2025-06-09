import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()
debug = os.getenv("DEBUG") == "true"

def simple_custom_logger(name):
    custom_logger = logging.getLogger(name)
    if debug: custom_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)

    # Set the logging level for the console handler
    console_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for the console handler
    formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    custom_logger.addHandler(console_handler)

    return custom_logger

logger = simple_custom_logger("TAS")