import os
import sys
import time
import multiprocessing
import atexit
import signal
from datetime import datetime
from tqdm import tqdm
# -------------------------------------------------------------------
# Common Log Directory
# -------------------------------------------------------------------
COMMON_LOG_DIR = os.environ.get("COMMON_LOG_DIR", os.path.join(os.path.expanduser("~"), "logs"))
os.makedirs(COMMON_LOG_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Terminal Color Codes (ANSI)
# -------------------------------------------------------------------
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    PURPLE = '\033[95m'
    WHITE = '\033[39m'
    ORIGINAL = '\033[39;49m'
    BLACK = '\033[30m'

    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'


# -------------------------------------------------------------------
# Helper function (ensuring positive results)
# -------------------------------------------------------------------
def _subtract(a: int, b: int) -> int:
    return a - b if a >= b else 0


# -------------------------------------------------------------------
# Animated Pointer
# -------------------------------------------------------------------
def run_pointer(speed: float, pointer_char: str, pointer_ref: str) -> None:
    """
    Continuously animates a pointer character back and forth over pointer_ref.
    """
    range_pointer_ref = range(1, len(pointer_ref))
    len_pointer_ref = len(pointer_ref) - 1
    velocity = 1 - speed

    def pointer_forward():
        for index in range_pointer_ref:
            print(f'|{pointer_ref[:index]}{pointer_char}{pointer_ref[index + 1:]}|', end='\r', flush=True)
            time.sleep(velocity)

    def pointer_backward():
        for i in range_pointer_ref:
            index = len_pointer_ref - i
            print(f'|{pointer_ref[:index]}{pointer_char}{pointer_ref[index + 1:]}|', end='\r', flush=True)
            time.sleep(velocity)

    try:
        while True:
            pointer_forward()
            time.sleep(velocity)
            pointer_backward()
            time.sleep(velocity)
    except Exception:
        return


class CustomLogger:
    def __init__(
            self,
            name: str = "CustomLogger",
            log_dir: str = None,
            speed: float = 0.937,
            time_format: str = "%Y-%m-%d %H:%M:%S",
            pointer_char: str = '⚮'
    ):
        """
        Initializes the custom logger.

        Parameters:
            name (str): Logger name. Also used for the log file name.
            log_dir (str): Directory to store the log file. Defaults to COMMON_LOG_DIR if not provided.
            speed (float): Speed for the pointer animation (0 < speed < 1).
            time_format (str): Date/time format for log entries.
            pointer_char (str): Character used in the animated pointer.
        """
        self.name = name
        self.time_format = time_format
        self.speed = speed
        self.pointer_char = pointer_char
        self.onflush = False
        self.pointer_process = None
        self.tqdm_bar = None

        # Determine logging directory
        if log_dir is None:
            log_dir = COMMON_LOG_DIR
        os.makedirs(log_dir, exist_ok=True)

        self.log_file_path = os.path.join(log_dir, f"{self.name}.log")

        try:
            self.log_file = open(self.log_file_path, 'a', encoding='utf-8')

        except Exception as e:
            print(f"Error opening log file: {e}")
            self.log_file = None

    def _write_file(self, text: str) -> None:
        """Write a single log entry to the log file, flush immediately."""
        if self.log_file:
            try:
                self.log_file.write(text + "\n")
                self.log_file.flush()
            except Exception as e:
                print(f"Error writing to log file: {e}")

    def _get_terminal_width(self) -> int:
        """Returns the width of the current terminal, or a fallback if unavailable."""
        try:
            return os.get_terminal_size().columns
        except OSError:
            return 250

    def _log(self, bg_color: str, fg_color: str, level: str, message: str, flush: bool = False) -> None:
        """
        Core log method that prints to the console with colors and adaptive wrapping,
        and writes a plain-text entry to the log file.
        """
        formatted_date_time = datetime.now().strftime(self.time_format)
        terminal_width = self._get_terminal_width()

        header = f"| {formatted_date_time} | :: | {bg_color}{fg_color}{level.ljust(7)}{bcolors.ENDC} | :: | "
        header_length = len(header)

        available_width = terminal_width - header_length - 2
        lines = [message[i:i + available_width] for i in range(0, len(message), available_width)]
        if not lines:
            lines = [""]

        if self.pointer_process is not None and self.pointer_process.is_alive():
            self.pointer_process.terminate()
            self.pointer_process = None

        full_line = header + lines[0] + " " * _subtract(terminal_width, header_length + len(lines[0]))
        print(full_line)

        for extra_line in lines[1:]:
            indent = " " * header_length
            print(indent + extra_line)

        file_entry = f"{formatted_date_time} | {level} | {message}"
        self._write_file(file_entry)

        if flush:
            self.onflush = True
            print()

    # -------------------------------------------------------------------
    # Public Logging Methods
    # -------------------------------------------------------------------
    def info(self, message: str, flush: bool = False) -> None:
        self._log(bcolors.BG_WHITE, bcolors.BLACK, 'INFO', message, flush)

    def ok(self, message: str, flush: bool = False) -> None:
        self._log(bcolors.BG_GREEN, bcolors.BLACK, 'OK', message, flush)

    def warning(self, message: str, flush: bool = False) -> None:
        self._log(bcolors.BG_YELLOW, bcolors.BLACK, 'WARNING', message, flush)

    def error(self, message: str, flush: bool = False) -> None:
        self._log(bcolors.BG_RED, bcolors.WHITE, 'ERROR', message, flush)

    def wrap_line(self, flush: bool = False) -> None:
        """
        Prints a blank line to the console and optionally inserts a blank line in the log file.
        """
        print()
        if flush:
            self._write_file("")

    def pointer(self) -> None:
        """
        Starts an animated pointer (spinner) in a separate process.
        This can be used to indicate that a long-running operation is in progress.
        """
        if self.tqdm_bar is not None:
            pass

        formatted_date_time = datetime.now().strftime(self.time_format)
        pointer_ref = " " * (len(formatted_date_time) + 10)

        self.pointer_process = multiprocessing.Process(
            target=run_pointer,
            args=(self.speed, self.pointer_char, pointer_ref)
        )
        self.pointer_process.daemon = True
        self.pointer_process.start()

    def progress_bar(self, iterable=None, total=None, desc=None, **tqdm_kwargs):
        """
        Creates and returns a TQDM progress bar. If a pointer is running,
        terminate it to avoid console conflicts.

        Usage:
            with logger.progress_bar(range(100), desc="Processing") as pbar:
                for item in pbar:
                    ...  # do something

        or:
            pbar = logger.progress_bar(total=100, desc="Processing")
            for i in range(100):
                ...  # do something
                pbar.update(1)
            pbar.close()

        Parameters:
            iterable: An optional iterable to wrap with tqdm.
            total: Total iterations (if no iterable, you must specify total).
            desc: Short description text in the progress bar.
            **tqdm_kwargs: Additional arguments you want to pass to tqdm (e.g., `leave=False`).
        """
        if self.pointer_process is not None and self.pointer_process.is_alive():
            self.pointer_process.terminate()
            self.pointer_process = None

        if self.tqdm_bar is not None:
            self.tqdm_bar.close()
            self.tqdm_bar = None

        self.tqdm_bar = tqdm(iterable=iterable, total=total, desc=desc, **tqdm_kwargs)
        return self.tqdm_bar

    def close_progress_bar(self):
        """
        Closes the tqdm progress bar if it is active.
        """
        if self.tqdm_bar is not None:
            self.tqdm_bar.close()
            self.tqdm_bar = None

    def close(self) -> None:
        """
        Close the log file and terminate any pointer process.
        Also close any active tqdm progress bar.
        """
        # Terminate pointer process
        if self.pointer_process is not None and self.pointer_process.is_alive():
            self.pointer_process.terminate()

        # Close TQDM bar
        if self.tqdm_bar is not None:
            self.tqdm_bar.close()
            self.tqdm_bar = None

        # Close log file
        if self.log_file:
            try:
                self.log_file.close()
            except Exception as e:
                print(f"Error closing log file: {e}")