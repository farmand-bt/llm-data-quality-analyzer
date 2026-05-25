import os
from unittest.mock import MagicMock, patch

import pytest

from narrator.client import generate, is_configured
from narrator.narrator import Narrator
from narrator.prompts import build_cleaning_prompt, build_column_prompt, build_overview_prompt


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def report():
    return {
        "dataset": {
            "name": "test.csv",
            "rows": 100,
            "columns": 2,
            "memory_mb": 0.1,
            "overall_missing_pct": 5.0,
            "quality_score": 75,
            "quality_grade": "C",
        },
        "columns": {
            "price": {
                "pandas_dtype": "float64",
                "inferred_type": "numeric",
                "type_mismatch": False,
                "missing_count": 5,
                "missing_pct": 5.0,
                "missing_pattern": "MCAR",
                "stats": {
                    "mean": 250000.0,
                    "median": 240000.0,
                    "std": 50000.0,
                    "min": 100000.0,
                    "max": 500000.0,
                },
                "outliers": {
                    "iqr_count": 2,
                    "iqr_pct": 2.0,
                    "iqr_indices": [10, 20],
                    "zscore_count": 1,
                    "zscore_pct": 1.0,
                    "zscore_indices": [10],
                },
                "distribution": {
                    "normality_pvalue": 0.05,
                    "is_normal": False,
                    "skew_label": "right",
                    "kurtosis_label": "normal",
                    "high_cardinality": None,
                    "datetime_gaps": None,
                },
                "warnings": ["5.0% missing values"],
            },
            "city": {
                "pandas_dtype": "object",
                "inferred_type": "categorical",
                "type_mismatch": False,
                "missing_count": 0,
                "missing_pct": 0.0,
                "missing_pattern": "none",
                "stats": {"unique_count": 10, "top_value": "Berlin"},
                "outliers": {
                    "iqr_count": 0,
                    "iqr_pct": 0.0,
                    "iqr_indices": [],
                    "zscore_count": 0,
                    "zscore_pct": 0.0,
                    "zscore_indices": [],
                },
                "distribution": {
                    "normality_pvalue": None,
                    "is_normal": None,
                    "skew_label": None,
                    "kurtosis_label": None,
                    "high_cardinality": False,
                    "datetime_gaps": None,
                },
                "warnings": [],
            },
        },
        "duplicates": {
            "exact_count": 3,
            "exact_pct": 3.0,
            "sample_indices": [1, 2],
            "near_duplicate_count": 0,
            "near_duplicate_pct": 0.0,
            "near_sample_indices": [],
        },
        "correlations": {"pearson": {}, "cramers_v": {}, "high_correlation_pairs": []},
        "recommendations": [
            {
                "severity": "high",
                "category": "missing",
                "column": "price",
                "message": "5.0% missing values",
            },
        ],
        "generated_at": "2026-05-25T10:00:00",
    }


# ---------------------------------------------------------------------------
# is_configured
# ---------------------------------------------------------------------------


class TestIsConfigured:
    def test_false_when_no_env_vars(self):
        with patch.dict(os.environ, {}, clear=True):
            assert not is_configured()

    def test_false_when_only_key_set(self):
        with patch.dict(os.environ, {"GWDG_API_KEY": "key"}, clear=True):
            assert not is_configured()

    def test_false_when_only_base_set(self):
        with patch.dict(os.environ, {"GWDG_API_BASE": "https://example.com"}, clear=True):
            assert not is_configured()

    def test_true_when_both_set(self):
        with patch.dict(
            os.environ,
            {"GWDG_API_KEY": "key", "GWDG_API_BASE": "https://example.com"},
        ):
            assert is_configured()


# ---------------------------------------------------------------------------
# generate (client)
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_raises_when_not_configured(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="not configured"):
                generate("hello")

    def test_returns_stripped_response_text(self):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "  Great data!  "

        with patch.dict(
            os.environ,
            {"GWDG_API_KEY": "k", "GWDG_API_BASE": "https://example.com"},
        ):
            with patch("narrator.client.OpenAI") as mock_openai:
                mock_openai.return_value.chat.completions.create.return_value = mock_response
                result = generate("test prompt", max_tokens=100)

        assert result == "Great data!"

    def test_passes_max_tokens_to_api(self):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "ok"

        with patch.dict(
            os.environ,
            {"GWDG_API_KEY": "k", "GWDG_API_BASE": "https://example.com"},
        ):
            with patch("narrator.client.OpenAI") as mock_openai:
                mock_openai.return_value.chat.completions.create.return_value = mock_response
                generate("prompt", max_tokens=42)
                call_kwargs = mock_openai.return_value.chat.completions.create.call_args[1]

        assert call_kwargs["max_tokens"] == 42

    def test_raises_on_auth_error(self):
        from openai import AuthenticationError

        http_response = MagicMock()
        http_response.request = MagicMock()
        http_response.status_code = 401
        http_response.json.return_value = {
            "error": {
                "message": "bad key",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            }
        }

        with patch.dict(
            os.environ,
            {"GWDG_API_KEY": "bad", "GWDG_API_BASE": "https://example.com"},
        ):
            with patch("narrator.client.OpenAI") as mock_openai:
                mock_openai.return_value.chat.completions.create.side_effect = (
                    AuthenticationError("bad key", response=http_response, body={})
                )
                with pytest.raises(RuntimeError, match="authentication failed"):
                    generate("test")

    def test_retries_on_rate_limit_then_succeeds(self):
        from openai import RateLimitError

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "ok"

        http_response = MagicMock()
        http_response.request = MagicMock()
        http_response.status_code = 429
        http_response.json.return_value = {
            "error": {"message": "rate limited", "type": "rate_limit_error", "code": None}
        }
        rate_error = RateLimitError("rate limited", response=http_response, body={})

        with patch.dict(
            os.environ,
            {"GWDG_API_KEY": "k", "GWDG_API_BASE": "https://example.com"},
        ):
            with patch("narrator.client.OpenAI") as mock_openai:
                with patch("narrator.client.time.sleep"):
                    mock_openai.return_value.chat.completions.create.side_effect = [
                        rate_error,
                        mock_response,
                    ]
                    result = generate("test")

        assert result == "ok"

    def test_raises_after_all_retries_exhausted(self):
        from openai import RateLimitError

        http_response = MagicMock()
        http_response.request = MagicMock()
        http_response.status_code = 429
        http_response.json.return_value = {
            "error": {"message": "rate limited", "type": "rate_limit_error", "code": None}
        }
        rate_error = RateLimitError("rate limited", response=http_response, body={})

        with patch.dict(
            os.environ,
            {"GWDG_API_KEY": "k", "GWDG_API_BASE": "https://example.com"},
        ):
            with patch("narrator.client.OpenAI") as mock_openai:
                with patch("narrator.client.time.sleep"):
                    mock_openai.return_value.chat.completions.create.side_effect = rate_error
                    with pytest.raises(RuntimeError, match="unavailable after"):
                        generate("test")


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


class TestBuildOverviewPrompt:
    def test_contains_dataset_name(self, report):
        assert "test.csv" in build_overview_prompt(report)

    def test_contains_row_count(self, report):
        assert "100" in build_overview_prompt(report)

    def test_lists_missing_column(self, report):
        assert "price" in build_overview_prompt(report)

    def test_shows_none_when_no_missing(self, report):
        for col in report["columns"].values():
            col["missing_count"] = 0
        assert "none" in build_overview_prompt(report)

    def test_shows_duplicate_count(self, report):
        assert "3" in build_overview_prompt(report)

    def test_quality_score_in_prompt(self, report):
        assert "75" in build_overview_prompt(report)

    def test_no_outliers_message(self, report):
        for col in report["columns"].values():
            col["outliers"]["iqr_count"] = 0
        assert "none detected" in build_overview_prompt(report)


class TestBuildColumnPrompt:
    def test_contains_column_name(self, report):
        assert "price" in build_column_prompt(report, "price")

    def test_contains_inferred_type(self, report):
        assert "numeric" in build_column_prompt(report, "price")

    def test_contains_missing_pct(self, report):
        assert "5.0" in build_column_prompt(report, "price")

    def test_contains_outlier_count(self, report):
        assert "2" in build_column_prompt(report, "price")

    def test_contains_mean_stat(self, report):
        assert "mean" in build_column_prompt(report, "price")

    def test_shows_no_warnings_for_clean_column(self, report):
        assert "none" in build_column_prompt(report, "city")

    def test_unknown_column_raises(self, report):
        with pytest.raises(KeyError):
            build_column_prompt(report, "nonexistent")


class TestBuildCleaningPrompt:
    def test_contains_recommendation_message(self, report):
        assert "5.0% missing values" in build_cleaning_prompt(report)

    def test_severity_label_uppercased(self, report):
        assert "HIGH" in build_cleaning_prompt(report)

    def test_column_tag_present(self, report):
        assert "[price]" in build_cleaning_prompt(report)

    def test_no_issues_message_when_empty(self, report):
        report["recommendations"] = []
        assert "No issues detected" in build_cleaning_prompt(report)


# ---------------------------------------------------------------------------
# Narrator class
# ---------------------------------------------------------------------------


class TestNarrator:
    def test_narrate_overview_returns_text(self, report):
        with patch("narrator.narrator.generate", return_value="overview text"):
            assert Narrator(report).narrate_overview() == "overview text"

    def test_narrate_overview_prompt_contains_dataset_name(self, report):
        with patch("narrator.narrator.generate", return_value="x") as mock_gen:
            Narrator(report).narrate_overview()
        assert "test.csv" in mock_gen.call_args[0][0]

    def test_narrate_column_returns_text(self, report):
        with patch("narrator.narrator.generate", return_value="col text"):
            assert Narrator(report).narrate_column("price") == "col text"

    def test_narrate_column_prompt_contains_col_name(self, report):
        with patch("narrator.narrator.generate", return_value="x") as mock_gen:
            Narrator(report).narrate_column("price")
        assert "price" in mock_gen.call_args[0][0]

    def test_narrate_cleaning_plan_returns_text(self, report):
        with patch("narrator.narrator.generate", return_value="clean text"):
            assert Narrator(report).narrate_cleaning_plan() == "clean text"

    def test_narrate_cleaning_plan_prompt_contains_severity(self, report):
        with patch("narrator.narrator.generate", return_value="x") as mock_gen:
            Narrator(report).narrate_cleaning_plan()
        assert "HIGH" in mock_gen.call_args[0][0]

    def test_overview_max_tokens(self, report):
        with patch("narrator.narrator.generate", return_value="x") as mock_gen:
            Narrator(report).narrate_overview()
        assert mock_gen.call_args[1]["max_tokens"] == 300

    def test_column_max_tokens(self, report):
        with patch("narrator.narrator.generate", return_value="x") as mock_gen:
            Narrator(report).narrate_column("price")
        assert mock_gen.call_args[1]["max_tokens"] == 200

    def test_cleaning_plan_max_tokens(self, report):
        with patch("narrator.narrator.generate", return_value="x") as mock_gen:
            Narrator(report).narrate_cleaning_plan()
        assert mock_gen.call_args[1]["max_tokens"] == 400

    def test_does_not_mutate_report(self, report):
        original_score = report["dataset"]["quality_score"]
        with patch("narrator.narrator.generate", return_value="x"):
            Narrator(report).narrate_overview()
        assert report["dataset"]["quality_score"] == original_score
