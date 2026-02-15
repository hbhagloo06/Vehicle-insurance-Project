import sys
import logging
from typing import Optional


def error_message_detail(error: BaseException, error_detail: Optional[object] = None) -> str:
    """
    Build a detailed error string: file name, line number, and original error message.
    """
    # Prefer the traceback attached to the exception (works even outside an except block)
    tb = error.__traceback__

    # Fallback: if caller passed sys and we're inside an except block
    if tb is None and error_detail is sys:
        _, _, tb = sys.exc_info()

    # If we still don't have a traceback, return a minimal message
    if tb is None:
        msg = f"Error occurred: {type(error).__name__}: {error}"
        logging.error(msg)
        return msg

    file_name = tb.tb_frame.f_code.co_filename
    line_number = tb.tb_lineno

    msg = (
        f"Error occurred in python script: [{file_name}] "
        f"at line number [{line_number}]: {type(error).__name__}: {error}"
    )
    logging.error(msg)
    return msg


class MyException(Exception):
    """
    Custom exception class with detailed context.
    """
    def __init__(self, error: BaseException, error_detail: Optional[object] = sys):
        # Initialize base Exception with a normal message (keeps default behavior)
        super().__init__(str(error))

        # Store detailed message
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self) -> str:
        return self.error_message
    
