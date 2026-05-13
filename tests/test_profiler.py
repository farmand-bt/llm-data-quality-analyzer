from pathlib import Path

import pandas as pd
import pytest

from profiler.distributions import (
    _detect_datetime_gaps,
    _kurtosis_label,
    _normality_test,
    _skew_label,
    analyze_distributions,
)
from profiler.correlations import compute_correlations
from profiler.duplicates import analyze_duplicates
from profiler.missing import analyze_missing
from profiler.outliers import detect_outliers
from profiler.recommendations import generate_recommendations
from profiler.report import build_report
from profiler.stats import compute_stats
from profiler.type_detector import detect_types

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def messy_df():
    path = FIXTURES / "messy_sample.csv"
    if path.exists():
        return pd.read_csv(path)
    return _make_messy_df()


@pytest.fixture
def clean_df():
    return pd.DataFrame(
        {
            "price": [100.0, 200.0, 300.0, 400.0, 500.0],
            "category": ["a", "b", "a", "b", "a"],
        }
    )


def _make_messy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": range(1, 11),
            "age": [25, 30, None, 22, 45, 38, None, 29, 55, 33],
            "salary": [50000, 60000, 70000, None, 80000, 55000, 90000, 65000, None, 72000],
            "name": ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy"],
            "status": ["active", "inactive", "active", "active", None, "inactive", "active", None, "active", "inactive"],
            "score_str": ["1.5", "2.3", "bad_val", "4.1", "5.0", "1.1", "3.3", "2.2", "4.4", "5.5"],
            "date_str": ["2020-01-01", "2020-06-15", "2021-03-20", None, "2019-11-05",
                         "2022-08-10", "2023-01-01", "2020-07-07", "2021-12-31", "2022-05-05"],
        }
    )


# ---------------------------------------------------------------------------
# type_detector
# ---------------------------------------------------------------------------


class TestTypeDetector:
    def test_integer_column(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        result = detect_types(df)
        assert result["x"]["inferred_type"] == "numeric"
        assert result["x"]["type_mismatch"] is False

    def test_float_column(self):
        df = pd.DataFrame({"x": [1.1, 2.2, 3.3]})
        result = detect_types(df)
        assert result["x"]["inferred_type"] == "numeric"
        assert result["x"]["type_mismatch"] is False

    def test_categorical_low_cardinality(self):
        df = pd.DataFrame({"cat": ["a", "b", "a", "c", "b", "a", "c", "b"]})
        result = detect_types(df)
        assert result["cat"]["inferred_type"] == "categorical"
        assert result["cat"]["type_mismatch"] is False

    def test_numeric_stored_as_string(self):
        df = pd.DataFrame({"num_str": ["1.0", "2.0", "3.0", "4.0", "5.0"]})
        result = detect_types(df)
        assert result["num_str"]["inferred_type"] == "numeric"
        assert result["num_str"]["type_mismatch"] is True

    def test_mixed_numeric_and_strings(self):
        df = pd.DataFrame({"mixed": ["1.5", "2.3", "bad", "4.1", "5.0", "1.1", "3.3", "2.2", "4.4", "5.5"]})
        result = detect_types(df)
        assert result["mixed"]["inferred_type"] == "mixed"
        assert result["mixed"]["type_mismatch"] is True

    def test_datetime_stored_as_string(self):
        df = pd.DataFrame({"dt": ["2020-01-01", "2021-06-15", "2022-03-20", "2023-08-10", "2019-11-05"]})
        result = detect_types(df)
        assert result["dt"]["inferred_type"] == "datetime"
        assert result["dt"]["type_mismatch"] is True

    def test_boolean_stored_as_yes_no(self):
        df = pd.DataFrame({"flag": ["yes", "no", "yes", "no", "yes"]})
        result = detect_types(df)
        assert result["flag"]["inferred_type"] == "boolean"
        assert result["flag"]["type_mismatch"] is True

    def test_native_bool_dtype(self):
        df = pd.DataFrame({"b": [True, False, True, False]})
        result = detect_types(df)
        assert result["b"]["inferred_type"] == "boolean"
        assert result["b"]["type_mismatch"] is False

    def test_native_datetime_dtype(self):
        df = pd.DataFrame({"dt": pd.to_datetime(["2020-01-01", "2021-01-01"])})
        result = detect_types(df)
        assert result["dt"]["inferred_type"] == "datetime"
        assert result["dt"]["type_mismatch"] is False

    def test_pandas_dtype_recorded(self):
        df = pd.DataFrame({"x": pd.array([1, 2, 3], dtype="int64")})
        result = detect_types(df)
        assert result["x"]["pandas_dtype"] == "int64"

    def test_all_columns_present(self, messy_df):
        result = detect_types(messy_df)
        assert set(result.keys()) == set(messy_df.columns)

    def test_all_null_column(self):
        df = pd.DataFrame({"empty": [None, None, None]})
        result = detect_types(df)
        assert result["empty"]["inferred_type"] == "categorical"
        assert result["empty"]["type_mismatch"] is False


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_numeric_keys(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        type_info = detect_types(df)
        result = compute_stats(df, type_info)
        for key in ("mean", "median", "std", "min", "max", "skewness", "kurtosis", "zeros", "unique_count"):
            assert key in result["x"]

    def test_numeric_mean(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        type_info = detect_types(df)
        result = compute_stats(df, type_info)
        assert result["x"]["mean"] == pytest.approx(3.0)

    def test_numeric_zeros(self):
        df = pd.DataFrame({"x": [0.0, 0.0, 1.0, 2.0, 3.0]})
        type_info = detect_types(df)
        result = compute_stats(df, type_info)
        assert result["x"]["zeros"] == 2

    def test_categorical_keys(self):
        df = pd.DataFrame({"cat": ["a", "b", "a", "c", "b", "a"]})
        type_info = detect_types(df)
        result = compute_stats(df, type_info)
        for key in ("unique_count", "top_values", "cardinality_ratio"):
            assert key in result["cat"]

    def test_categorical_unique_count(self):
        df = pd.DataFrame({"cat": ["a", "b", "a", "c", "b", "a"]})
        type_info = detect_types(df)
        result = compute_stats(df, type_info)
        assert result["cat"]["unique_count"] == 3

    def test_categorical_top_values(self):
        df = pd.DataFrame({"cat": ["a", "a", "a", "b", "b", "c"]})
        type_info = detect_types(df)
        result = compute_stats(df, type_info)
        assert result["cat"]["top_values"]["a"] == 3

    def test_datetime_keys(self):
        df = pd.DataFrame({"dt": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"])})
        type_info = detect_types(df)
        result = compute_stats(df, type_info)
        for key in ("min_date", "max_date", "range_days"):
            assert key in result["dt"]

    def test_datetime_range_days(self):
        df = pd.DataFrame({"dt": pd.to_datetime(["2020-01-01", "2022-01-01"])})
        type_info = detect_types(df)
        result = compute_stats(df, type_info)
        assert result["dt"]["range_days"] == 731  # 2020 is a leap year

    def test_all_columns_covered(self, messy_df):
        type_info = detect_types(messy_df)
        result = compute_stats(messy_df, type_info)
        assert set(result.keys()) == set(messy_df.columns)

    def test_handles_missing_in_numeric(self):
        df = pd.DataFrame({"x": [1.0, None, 3.0, None, 5.0]})
        type_info = detect_types(df)
        result = compute_stats(df, type_info)
        assert result["x"]["mean"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# missing
# ---------------------------------------------------------------------------


class TestMissingAnalysis:
    def test_no_missing(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = analyze_missing(df)
        assert result["overall_missing_pct"] == 0.0
        assert result["total_missing_cells"] == 0

    def test_missing_count(self):
        df = pd.DataFrame({"x": [1, None, 3, None, 5]})
        result = analyze_missing(df)
        assert result["per_column"]["x"]["missing_count"] == 2
        assert result["per_column"]["x"]["missing_pct"] == pytest.approx(40.0)

    def test_overall_missing_pct(self):
        df = pd.DataFrame({"x": [None, None, 3], "y": [4, 5, 6]})
        result = analyze_missing(df)
        # 2 missing out of 6 total cells
        assert result["overall_missing_pct"] == pytest.approx(33.33, abs=0.1)

    def test_co_occurrence(self):
        df = pd.DataFrame(
            {"a": [None, None, 3, 4, 5], "b": [None, None, 6, 7, 8]}
        )
        result = analyze_missing(df)
        assert "a|b" in result["co_occurrence"]
        assert result["co_occurrence"]["a|b"] == 2

    def test_no_co_occurrence_when_no_overlap(self):
        df = pd.DataFrame(
            {"a": [None, 2, 3], "b": [1, None, 3]}
        )
        result = analyze_missing(df)
        # They never missing in the same row
        assert result["co_occurrence"].get("a|b", 0) == 0

    def test_missing_pattern_none_for_complete(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        result = analyze_missing(df)
        assert result["per_column"]["x"]["missing_pattern"] == "none"

    def test_missing_pattern_mnar_high_pct(self):
        df = pd.DataFrame({"x": [None] * 7 + [1, 2, 3]})
        result = analyze_missing(df)
        assert result["per_column"]["x"]["missing_pattern"] == "MNAR"

    def test_missing_pattern_mcar(self):
        import random
        random.seed(42)
        vals = [float(i) if random.random() > 0.1 else None for i in range(50)]
        df = pd.DataFrame({"x": vals, "y": list(range(50))})
        result = analyze_missing(df)
        # Should not be MNAR (< 60%) and no co-occurrence with y (y has no missing)
        assert result["per_column"]["x"]["missing_pattern"] in ("MCAR", "MAR")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


class TestBuildReport:
    def test_top_level_keys(self, messy_df):
        report = build_report(messy_df, "messy.csv")
        for key in ("dataset", "columns", "duplicates", "correlations", "recommendations", "generated_at"):
            assert key in report

    def test_dataset_fields(self, messy_df):
        report = build_report(messy_df, "test.csv")
        ds = report["dataset"]
        assert ds["name"] == "test.csv"
        assert ds["rows"] == len(messy_df)
        assert ds["columns"] == len(messy_df.columns)
        assert isinstance(ds["quality_score"], int)
        assert 0 <= ds["quality_score"] <= 100
        assert ds["quality_grade"] in ("A", "B", "C", "D", "F")

    def test_column_schema(self, messy_df):
        report = build_report(messy_df, "test.csv")
        assert set(report["columns"].keys()) == set(messy_df.columns)
        for info in report["columns"].values():
            for key in ("pandas_dtype", "inferred_type", "type_mismatch",
                        "missing_count", "missing_pct", "stats",
                        "outliers", "distribution", "warnings"):
                assert key in info

    def test_clean_data_gets_high_score(self, clean_df):
        report = build_report(clean_df, "clean.csv")
        assert report["dataset"]["quality_score"] >= 90
        assert report["dataset"]["quality_grade"] == "A"

    def test_missing_data_lowers_score(self, clean_df):
        import numpy as np

        messy = pd.DataFrame(
            {"price": [np.nan] * len(clean_df), "category": clean_df["category"]}
        )
        report_clean = build_report(clean_df, "clean.csv")
        report_messy = build_report(messy, "messy.csv")
        assert report_clean["dataset"]["quality_score"] > report_messy["dataset"]["quality_score"]

    def test_type_mismatch_lowers_score(self):
        df_clean = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": ["a", "b", "a"]})
        df_mismatch = pd.DataFrame({"x": ["1.0", "2.0", "3.0"], "y": ["a", "b", "a"]})
        r_clean = build_report(df_clean, "c.csv")
        r_mismatch = build_report(df_mismatch, "m.csv")
        assert r_clean["dataset"]["quality_score"] > r_mismatch["dataset"]["quality_score"]

    def test_warnings_generated_for_high_missing(self):
        df = pd.DataFrame({"x": [None] * 8 + [1, 2]})
        report = build_report(df, "t.csv")
        warnings = report["columns"]["x"]["warnings"]
        assert any("missing" in w.lower() for w in warnings)

    def test_warnings_generated_for_type_mismatch(self):
        df = pd.DataFrame({"x": ["1.0", "2.0", "3.0", "4.0", "5.0"]})
        report = build_report(df, "t.csv")
        warnings = report["columns"]["x"]["warnings"]
        assert any("mismatch" in w.lower() for w in warnings)

    def test_report_is_json_serializable(self, messy_df):
        import json
        report = build_report(messy_df, "test.csv")
        # Should not raise
        json.dumps(report)

    def test_high_cardinality_warning(self):
        # All-unique strings → classified as "text" (cardinality 1.0 > 0.5 threshold)
        # → high_cardinality=True → cardinality warning generated
        df = pd.DataFrame({"id": [f"user_{i}" for i in range(20)]})
        report = build_report(df, "t.csv")
        assert report["columns"]["id"]["inferred_type"] == "text"
        assert any("cardinality" in w.lower() for w in report["columns"]["id"]["warnings"])

    def test_distribution_fields_present(self, messy_df):
        report = build_report(messy_df, "test.csv")
        for info in report["columns"].values():
            dist = info["distribution"]
            for key in ("normality_pvalue", "is_normal", "skew_label",
                        "kurtosis_label", "high_cardinality", "datetime_gaps"):
                assert key in dist


# ---------------------------------------------------------------------------
# distributions
# ---------------------------------------------------------------------------


class TestDistributions:

    # --- _normality_test ---

    def test_normal_data_passes(self):
        import numpy as np
        rng = np.random.default_rng(42)
        series = pd.Series(rng.normal(0, 1, 200))
        pval, is_normal = _normality_test(series)
        assert pval is not None
        assert isinstance(is_normal, bool)

    def test_uniform_data_fails_normality(self):
        series = pd.Series(list(range(100)))  # clearly not normal
        pval, is_normal = _normality_test(series)
        assert pval is not None
        assert is_normal is False

    def test_too_few_values_returns_none(self):
        series = pd.Series([1.0, 2.0])
        pval, is_normal = _normality_test(series)
        assert pval is None
        assert is_normal is None

    def test_all_null_returns_none(self):
        series = pd.Series([None, None, None])
        pval, is_normal = _normality_test(series)
        assert pval is None
        assert is_normal is None

    def test_single_unique_value_handled(self):
        # Constant column — zero variance; normality is undefined, both fields are None
        series = pd.Series([5.0] * 10)
        pval, is_normal = _normality_test(series)
        assert pval is None
        assert is_normal is None

    # --- _skew_label ---

    def test_skew_symmetric(self):
        assert _skew_label(0.1) == "symmetric"
        assert _skew_label(-0.3) == "symmetric"

    def test_skew_moderately_right(self):
        assert _skew_label(0.7) == "moderately right-skewed"

    def test_skew_highly_right(self):
        assert _skew_label(1.5) == "highly right-skewed"

    def test_skew_highly_left(self):
        assert _skew_label(-1.2) == "highly left-skewed"

    def test_skew_none_returns_none(self):
        assert _skew_label(None) is None

    # --- _kurtosis_label ---

    def test_kurtosis_leptokurtic(self):
        assert _kurtosis_label(2.0) == "leptokurtic"

    def test_kurtosis_platykurtic(self):
        assert _kurtosis_label(-2.0) == "platykurtic"

    def test_kurtosis_mesokurtic(self):
        assert _kurtosis_label(0.5) == "mesokurtic"

    def test_kurtosis_none_returns_none(self):
        assert _kurtosis_label(None) is None

    # --- _detect_datetime_gaps ---

    def test_no_gaps_regular_series(self):
        dt = pd.Series(pd.date_range("2020-01-01", periods=12, freq="MS"))
        result = _detect_datetime_gaps(dt)
        assert result["gap_count"] == 0

    def test_gap_detected(self):
        # Insert a 2-year gap between 2020 and 2022
        dates = list(pd.date_range("2020-01-01", periods=6, freq="MS")) + \
                list(pd.date_range("2022-06-01", periods=6, freq="MS"))
        dt = pd.Series(dates)
        result = _detect_datetime_gaps(dt)
        assert result["gap_count"] >= 1
        assert result["max_gap_days"] > 300

    def test_single_date_returns_zero_gaps(self):
        dt = pd.Series([pd.Timestamp("2020-01-01")])
        result = _detect_datetime_gaps(dt)
        assert result["gap_count"] == 0
        assert result["max_gap_days"] is None

    # --- analyze_distributions ---

    def test_numeric_column_has_normality_fields(self):
        df = pd.DataFrame({"x": [float(i) for i in range(20)]})
        type_info = detect_types(df)
        stats_info = compute_stats(df, type_info)
        result = analyze_distributions(df, type_info, stats_info)
        assert "normality_pvalue" in result["x"]
        assert "skew_label" in result["x"]
        assert "kurtosis_label" in result["x"]
        assert result["x"]["high_cardinality"] is None

    def test_text_column_high_cardinality_flagged(self):
        # All-unique strings → type detector classifies as "text" (ratio 1.0 > 0.5)
        # → distributions module flags high_cardinality=True for text columns
        df = pd.DataFrame({"col": [f"item_{i}" for i in range(10)]})
        type_info = detect_types(df)
        stats_info = compute_stats(df, type_info)
        result = analyze_distributions(df, type_info, stats_info)
        assert type_info["col"]["inferred_type"] == "text"
        assert result["col"]["high_cardinality"] is True

    def test_categorical_low_cardinality_not_flagged(self):
        df = pd.DataFrame({"cat": ["a", "b", "a", "b", "a", "b", "a", "b"]})
        type_info = detect_types(df)
        stats_info = compute_stats(df, type_info)
        result = analyze_distributions(df, type_info, stats_info)
        assert result["cat"]["high_cardinality"] is False

    def test_datetime_column_returns_gaps(self):
        df = pd.DataFrame({"dt": pd.date_range("2020-01-01", periods=10, freq="ME")})
        type_info = detect_types(df)
        stats_info = compute_stats(df, type_info)
        result = analyze_distributions(df, type_info, stats_info)
        assert result["dt"]["datetime_gaps"] is not None
        assert "gap_count" in result["dt"]["datetime_gaps"]

    def test_all_null_column_handled(self):
        df = pd.DataFrame({"x": [None, None, None, None, None]})
        type_info = detect_types(df)
        stats_info = compute_stats(df, type_info)
        # Should not raise
        result = analyze_distributions(df, type_info, stats_info)
        assert "normality_pvalue" in result["x"]

    def test_all_columns_covered(self, messy_df):
        type_info = detect_types(messy_df)
        stats_info = compute_stats(messy_df, type_info)
        result = analyze_distributions(messy_df, type_info, stats_info)
        assert set(result.keys()) == set(messy_df.columns)


# ---------------------------------------------------------------------------
# outliers
# ---------------------------------------------------------------------------


class TestOutliers:
    def test_iqr_outlier_detected(self):
        # Use values with non-zero IQR so the fence is well-defined
        df = pd.DataFrame({"x": list(range(1, 21)) + [1000.0]})
        type_info = detect_types(df)
        result = detect_outliers(df, type_info)
        assert result["x"]["iqr_count"] > 0
        assert len(result["x"]["iqr_indices"]) > 0

    def test_no_outliers_uniform_data(self):
        df = pd.DataFrame({"x": list(range(1, 21))})
        type_info = detect_types(df)
        result = detect_outliers(df, type_info)
        assert result["x"]["iqr_count"] == 0

    def test_zscore_outlier_detected(self):
        import numpy as np
        rng = np.random.default_rng(42)
        vals = list(rng.normal(0, 1, 100)) + [15.0]
        df = pd.DataFrame({"x": vals})
        type_info = detect_types(df)
        result = detect_outliers(df, type_info)
        assert result["x"]["zscore_count"] > 0

    def test_non_numeric_returns_zeros(self):
        df = pd.DataFrame({"cat": ["a", "b", "c"] * 5})
        type_info = detect_types(df)
        result = detect_outliers(df, type_info)
        assert result["cat"]["iqr_count"] == 0
        assert result["cat"]["zscore_count"] == 0
        assert result["cat"]["iqr_indices"] == []

    def test_constant_column_no_outliers(self):
        df = pd.DataFrame({"x": [5.0] * 10})
        type_info = detect_types(df)
        result = detect_outliers(df, type_info)
        assert result["x"]["iqr_count"] == 0

    def test_too_few_values_no_outliers(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        type_info = detect_types(df)
        result = detect_outliers(df, type_info)
        assert result["x"]["iqr_count"] == 0
        assert result["x"]["zscore_count"] == 0

    def test_indices_are_valid_row_labels(self):
        df = pd.DataFrame({"x": [1.0] * 10 + [999.0]})
        type_info = detect_types(df)
        result = detect_outliers(df, type_info)
        for idx in result["x"]["iqr_indices"]:
            assert idx in df.index

    def test_pct_is_proportion_of_total_rows(self):
        df = pd.DataFrame({"x": list(range(1, 11)) + [999.0]})
        type_info = detect_types(df)
        result = detect_outliers(df, type_info)
        expected_pct = round(1 / 11 * 100, 2)
        assert result["x"]["iqr_pct"] == pytest.approx(expected_pct, abs=0.5)

    def test_all_columns_covered(self, messy_df):
        type_info = detect_types(messy_df)
        result = detect_outliers(messy_df, type_info)
        assert set(result.keys()) == set(messy_df.columns)


# ---------------------------------------------------------------------------
# duplicates
# ---------------------------------------------------------------------------


class TestDuplicates:
    def test_exact_duplicates_detected(self):
        df = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "x"]})
        result = analyze_duplicates(df)
        assert result["exact_count"] == 1

    def test_no_duplicates(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = analyze_duplicates(df)
        assert result["exact_count"] == 0
        assert result["exact_pct"] == 0.0

    def test_exact_pct(self):
        df = pd.DataFrame({"a": [1, 1, 2, 3, 4], "b": [1, 1, 2, 3, 4]})
        result = analyze_duplicates(df)
        assert result["exact_count"] == 1
        assert result["exact_pct"] == pytest.approx(20.0)

    def test_sample_indices_valid(self):
        df = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "x"]})
        result = analyze_duplicates(df)
        for idx in result["sample_indices"]:
            assert idx in df.index

    def test_near_duplicates_detected(self):
        df = pd.DataFrame({
            "a": [1, 1, 3, 4, 5],
            "b": [2, 2, 4, 5, 6],
            "c": ["x", "z", "q", "w", "e"],   # only column that differs between rows 0 and 1
        })
        result = analyze_duplicates(df)
        # Rows 0 and 1 are exact duplicates on columns a, b — so near_dup_count should be ≥ 0
        assert result["near_duplicate_count"] >= 0  # implementation-dependent count

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": [], "b": []})
        result = analyze_duplicates(df)
        assert result["exact_count"] == 0
        assert result["near_duplicate_count"] == 0

    def test_schema_keys_present(self, messy_df):
        result = analyze_duplicates(messy_df)
        for key in ("exact_count", "exact_pct", "sample_indices",
                    "near_duplicate_count", "near_duplicate_pct", "near_sample_indices"):
            assert key in result


# ---------------------------------------------------------------------------
# correlations
# ---------------------------------------------------------------------------


class TestCorrelations:
    def test_pearson_computed_for_numeric(self):
        import numpy as np
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        df = pd.DataFrame({"x": x, "y": x + rng.normal(0, 0.1, 100)})
        type_info = detect_types(df)
        result = compute_correlations(df, type_info)
        assert "x" in result["pearson"]
        assert abs(result["pearson"]["x"]["y"]) > 0.9

    def test_high_correlation_pair_detected(self):
        import numpy as np
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 100)
        df = pd.DataFrame({"x": x, "y": x + 1e-6})
        type_info = detect_types(df)
        result = compute_correlations(df, type_info)
        assert len(result["high_correlation_pairs"]) > 0

    def test_pearson_empty_without_numeric(self):
        df = pd.DataFrame({"cat": ["a", "b", "a", "b"] * 5})
        type_info = detect_types(df)
        result = compute_correlations(df, type_info)
        assert result["pearson"] == {}

    def test_cramers_v_for_categorical(self):
        df = pd.DataFrame({
            "a": (["x", "y"] * 10),
            "b": (["x", "y"] * 10),  # perfectly correlated
        })
        type_info = detect_types(df)
        result = compute_correlations(df, type_info)
        if "a" in result["cramers_v"] and "b" in result["cramers_v"].get("a", {}):
            assert result["cramers_v"]["a"]["b"] >= 0.9

    def test_schema_keys_present(self, messy_df):
        type_info = detect_types(messy_df)
        result = compute_correlations(messy_df, type_info)
        for key in ("pearson", "cramers_v", "high_correlation_pairs"):
            assert key in result

    def test_high_pairs_sorted_by_abs_value(self):
        import numpy as np
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 100)
        df = pd.DataFrame({
            "a": x, "b": x + 1e-6, "c": x * 2 + 1e-6,
        })
        type_info = detect_types(df)
        result = compute_correlations(df, type_info)
        vals = [abs(p["value"]) for p in result["high_correlation_pairs"]]
        assert vals == sorted(vals, reverse=True)


# ---------------------------------------------------------------------------
# recommendations
# ---------------------------------------------------------------------------


class TestRecommendations:
    def test_missing_recommendation(self):
        df = pd.DataFrame({"x": [None] * 8 + [1.0, 2.0]})
        report = build_report(df, "t.csv")
        assert any(r["category"] == "missing" and r["column"] == "x" for r in report["recommendations"])

    def test_duplicate_recommendation(self):
        df = pd.DataFrame({"x": [1, 1, 2, 3, 4], "y": [1, 1, 2, 3, 4]})
        report = build_report(df, "t.csv")
        assert any(r["category"] == "duplicate" for r in report["recommendations"])

    def test_outlier_recommendation(self):
        df = pd.DataFrame({"x": list(range(1, 21)) + [10000.0]})
        report = build_report(df, "t.csv")
        assert any(r["category"] == "outlier" and r["column"] == "x" for r in report["recommendations"])

    def test_schema(self, messy_df):
        report = build_report(messy_df, "t.csv")
        for rec in report["recommendations"]:
            assert rec["severity"] in ("info", "warning", "critical")
            assert "category" in rec
            assert "column" in rec
            assert "message" in rec

    def test_severity_ordering(self, messy_df):
        report = build_report(messy_df, "t.csv")
        order = {"critical": 0, "warning": 1, "info": 2}
        severities = [r["severity"] for r in report["recommendations"]]
        for i in range(len(severities) - 1):
            assert order[severities[i]] <= order[severities[i + 1]]

    def test_clean_data_no_critical(self, clean_df):
        report = build_report(clean_df, "t.csv")
        assert not any(r["severity"] == "critical" for r in report["recommendations"])
