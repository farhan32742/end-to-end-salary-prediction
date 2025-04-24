import sys

from src.logger import logging
def error_message_details():
    pass


class CustomException(Exception):
    """Custom base exception for application-specific errors."""
    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,error_details)


    def __str__(self):import sys


def error_message_details(error_message, error_details: sys):
    """
    Extracts and formats error details including file name, line number, and function name.

    Args:
        error_message (str): The custom error message.
        error_details (sys): The sys module to extract exception info.

    Returns:
        str: A formatted string with detailed error information.
    """
    _, _, exc_tb = error_details.exc_info()
    tb = exc_tb.tb_frame if exc_tb else None

    if tb:
        file_name = tb.f_code.co_filename
        line_number = exc_tb.tb_lineno
        formatted_message = (
            f"Error: {error_message} | "
            f"File: {file_name} | "
            f"Line: {line_number}"
        )
    else:
        formatted_message = f"Error: {error_message} (No traceback available)"

    return formatted_message


class CustomException(Exception):
    """Custom base exception for application-specific errors."""
    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_details)

    def __str__(self):
        return self.error_message

        



