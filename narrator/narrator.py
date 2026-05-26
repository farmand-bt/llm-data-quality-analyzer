from narrator.client import generate
from narrator.prompts import (
    build_cleaning_prompt,
    build_column_prompt,
    build_overview_prompt,
    build_question_prompt,
)


class Narrator:
    """Generates plain-English narratives from a profiler report dict.

    The report dict is the only input — no raw DataFrame rows are ever used.
    """

    def __init__(self, report: dict) -> None:
        self._report = report

    def narrate_overview(self) -> str:
        return generate(build_overview_prompt(self._report), max_tokens=300)

    def narrate_column(self, col_name: str) -> str:
        return generate(build_column_prompt(self._report, col_name), max_tokens=200)

    def narrate_cleaning_plan(self) -> str:
        return generate(build_cleaning_prompt(self._report), max_tokens=600)

    def answer_question(self, question: str) -> str:
        return generate(build_question_prompt(self._report, question), max_tokens=400)
