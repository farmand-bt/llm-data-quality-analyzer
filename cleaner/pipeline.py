import pandas as pd

from cleaner.duplicate_handler import remove_duplicates
from cleaner.missing_handler import handle_missing
from cleaner.outlier_handler import handle_outliers
from cleaner.type_fixer import fix_types


class CleaningPipeline:
    """Queues and executes cleaning operations non-destructively.

    Usage:
        pipeline = CleaningPipeline()
        pipeline.add_step({"action": "handle_missing", "column": "age", "strategy": "fill_median"})
        pipeline.add_step({"action": "remove_duplicates", "keep": "first"})
        cleaned_df = pipeline.execute(original_df)
        log = pipeline.get_log()
    """

    def __init__(self) -> None:
        self._steps: list[dict] = []
        self._log: list[dict] = []

    def add_step(self, operation: dict) -> None:
        """Queue a cleaning step.

        Required key: "action"
        Per-action required keys:
            handle_missing:   column, strategy, [value]
            remove_duplicates: [keep]
            fix_type:         column, target_type
            handle_outliers:  column, indices, strategy
        """
        self._steps.append(operation)

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all queued steps sequentially. Original df is never modified."""
        result = df.copy()
        self._log = []

        for step in self._steps:
            action = step["action"]

            if action == "handle_missing":
                result, entry = handle_missing(
                    result,
                    step["strategy"],
                    step["column"],
                    value=step.get("value"),
                )
            elif action == "remove_duplicates":
                result, entry = remove_duplicates(result, keep=step.get("keep", "first"))
            elif action == "fix_type":
                result, entry = fix_types(result, step["column"], step["target_type"])
            elif action == "handle_outliers":
                result, entry = handle_outliers(
                    result,
                    step["column"],
                    step["indices"],
                    step["strategy"],
                )
            else:
                raise ValueError(f"Unknown action '{action}'")

            self._log.append(entry)

        return result

    def get_log(self) -> list[dict]:
        """Return the log of all applied transformations (populated after execute)."""
        return list(self._log)

    def get_steps(self) -> list[dict]:
        """Return the queued steps."""
        return list(self._steps)

    def clear(self) -> None:
        """Remove all queued steps and clear the log."""
        self._steps = []
        self._log = []

    def __len__(self) -> int:
        return len(self._steps)
