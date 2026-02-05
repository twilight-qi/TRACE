import logging
import os
from typing import Mapping, Optional

import rootutils
from colorlog import ColoredFormatter
from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only


class RelativePathFormatter(ColoredFormatter):
    """A custom formatter to show relative file paths in logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Calculates the relative path of the file and adds it to the log record."""
        # Heuristic to find the project root (assuming this file is in 'src/utils/')
        project_root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
        relative_path = os.path.relpath(record.pathname, project_root)
        record.relativePath = f"{relative_path}:{record.lineno}"
        return super().format(record)


class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Initializes a multi-GPU-friendly python command line logger that logs on all processes
        with their rank prefixed in the log message.

        :param name: The name of the logger. Default is ``__name__``.
        :param rank_zero_only: Whether to force all logs to only occur on the rank zero process. Default is `False`.
        :param extra: (Optional) A dict-like object which provides contextual information. See `logging.LoggerAdapter`.
        """
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: str, rank: Optional[int] = None, *args, **kwargs) -> None:
        """Delegate a log call to the underlying logger, after prefixing its message with the rank
        of the process it's being logged from. If `'rank'` is provided, then the log will only
        occur on that rank/process.

        :param level: The level to log at. Look at `logging.__init__.py` for more information.
        :param msg: The message to log.
        :param rank: The rank to log at.
        :param args: Additional args to pass to the underlying logging function.
        :param kwargs: Any additional keyword args to pass to the underlying logging function.
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
            msg = rank_prefixed_message(msg, current_rank)
            if self.rank_zero_only:
                if current_rank == 0:
                    # When `stacklevel=2` is used, the logging framework looks two frames up the call stack
                    # to find the caller's information (file name, line number).
                    #
                    # Call stack:
                    # 1. Your code: `log.info("Hello")` in `src/utils/utils.py`
                    # 2. This method: `RankedLogger.log(...)` in `src/utils/pylogger.py`
                    # 3. The actual logging call: `self.logger.log(...)`
                    #
                    # - `stacklevel=1` (default): Would report from frame 3, showing `pylogger.py` and the line below.
                    # - `stacklevel=2`: Reports from frame 2's caller (frame 1), correctly showing `src/utils/utils.py`
                    #   and the line where `log.info` was called.
                    self.logger.log(level, msg, *args, stacklevel=2, **kwargs)
            else:
                if rank is None:
                    self.logger.log(level, msg, *args, stacklevel=2, **kwargs)
                elif current_rank == rank:
                    self.logger.log(level, msg, *args, stacklevel=2, **kwargs)
