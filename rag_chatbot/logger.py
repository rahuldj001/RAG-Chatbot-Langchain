"""Logger utility that intercepts sys.stdout and tees it to a log file.

Allows the Gradio UI's 'Output' tab to display real-time terminal progress
and LangChain reasoning logs directly in the browser.
"""

import os
import sys
import re


class Logger:
    def __init__(self, filename: str):
        # Resolve full path to the log file in current working directory
        self.filename = os.path.join(os.getcwd(), filename)
        # Preserve original stdout so console printing continues working
        self.terminal = sys.stdout
        # Clear any existing log content from previous runs
        self.reset_logs()
        # Open the file for writing intercepted logs
        self.log = open(self.filename, "w", encoding="utf-8")
        self.flush()

    def write(self, message: str):
        """Tees output to both the terminal console and the disk log file."""
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """Flushes buffers to ensure log entries are immediately flushed to disk."""
        self.terminal.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return False

    def reset_logs(self):
        """Truncates the log file to start with a clean slate."""
        with open(self.filename, "w", encoding="utf-8") as file:
            file.truncate(0)

    def read_logs(self) -> str:
        """Reads and formats recent log entries for periodic display in the Gradio UI."""
        sys.stdout.flush()

        # Read all recorded log lines from disk
        with open(self.filename, "r", encoding="utf-8", errors="ignore") as f:
            log_content = f.readlines()

        # Sanitize lines by filtering out null bytes
        log_content = [line for line in log_content if "\x00" not in line]

        # Pattern to detect tqdm / progress bar lines
        progress_pattern = re.compile(r"\[.*\] \d+\.\d+%")

        # Extract active progress bar update lines
        progress_lines = [
            line
            for line in log_content
            if progress_pattern.search(line) and " - Completed!\n" not in line
        ]

        # Deduplicate repeated progress bars so they don't clutter the UI view
        if progress_lines:
            valid_content = [line for line in log_content if line not in progress_lines]
            if log_content[-1] == progress_lines[-1]:
                valid_content.append(progress_lines[-1].strip("\n"))
        else:
            valid_content = log_content

        # Return the most recent 300 lines for the UI Output tab
        recent_lines = valid_content[-300:]
        return "".join(recent_lines)
