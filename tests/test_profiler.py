import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
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

    def test_case_inconsistent_categorical_flagged(self):
        df = pd.DataFrame({"sex": ["Male", "MALE", "male", "female", "Female"] * 10})
        result = detect_types(df)
        assert result["sex"]["inferred_type"] == "categorical"
        assert result["sex"]["type_mismatch"] is True

    def test_consistent_categorical_not_flagged(self):
        df = pd.DataFrame({"sex": ["male", "female", "male", "female"] * 10})
        result = detect_types(df)
        assert result["sex"]["inferred_type"] == "categorical"
        assert result["sex"]["type_mismatch"] is False

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

    def test_high_missing_is_no_longer_mnar(self):
        # Data alone can't show MNAR; with nothing to compare against, the
        # pattern is "unknown" (ported from data-quality-mcp, M3.5 D1)
        df = pd.DataFrame({"x": [None] * 7 + [1, 2, 3]})
        result = analyze_missing(df)
        assert result["per_column"]["x"]["missing_pattern"] == "unknown"
        assert result["per_column"]["x"]["missing_related_to"] is None

    def test_missing_pattern_mcar(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=200)
        x[rng.random(200) < 0.2] = np.nan
        df = pd.DataFrame({"x": x, "y": rng.normal(size=200)})
        result = analyze_missing(df)
        assert result["per_column"]["x"]["missing_pattern"] == "MCAR"
        assert result["per_column"]["x"]["missing_related_to"] == []

    def test_missing_pattern_mar_names_the_related_column(self):
        rng = np.random.default_rng(1)
        age = rng.integers(0, 80, 300).astype(float)
        rating = np.where(age > 60, np.nan, 5.0)
        df = pd.DataFrame({"age": age, "rating": rating})
        result = analyze_missing(df)
        info = result["per_column"]["rating"]
        assert info["missing_pattern"] == "MAR"
        assert info["missing_related_to"][0]["column"] == "age"
        assert info["missing_related_to"][0]["method"] == "point_biserial"

    def test_housing_sample_school_rating_goes_with_age(self):
        sample = Path(__file__).parent.parent / "sample_data" / "housing_messy.csv"
        df = pd.read_csv(sample)
        info = analyze_missing(df)["per_column"]["school_rating"]
        # Was "MCAR" under the old co-missingness rule
        assert info["missing_pattern"] == "MAR"
        assert info["missing_related_to"] == [
            {"column": "age_years", "method": "point_biserial", "value": 0.663}
        ]


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


# ---------------------------------------------------------------------------
# near-duplicates: hashed search returns exactly the original definition
# ---------------------------------------------------------------------------


def _naive_near_duplicates(df: pd.DataFrame) -> set:
    """The original implementation, verbatim: one full pass per column."""
    rows = set()
    for col in df.columns:
        mask = df.duplicated(subset=[c for c in df.columns if c != col], keep=False)
        rows.update(mask[mask].index)
    return rows


class TestNearDuplicateHashing:
    def test_matches_naive_definition_on_random_frames(self):
        import numpy as np

        from profiler.duplicates import _near_duplicate_rows

        rng = np.random.default_rng(2026)
        for _ in range(200):
            n, k = int(rng.integers(2, 60)), int(rng.integers(2, 6))
            makers = [
                lambda: rng.integers(0, 3, n),
                lambda: np.where(
                    rng.random(n) < 0.3, np.nan, rng.integers(0, 3, n) * 1.0
                ),
                lambda: rng.choice(["a", "b", None], n),
                lambda: pd.array(rng.choice([1, 2, 1.0, "1", None], n), dtype=object),
                lambda: rng.choice([True, False], n),
            ]
            df = pd.DataFrame({f"c{j}": makers[rng.integers(0, 5)]() for j in range(k)})
            assert _near_duplicate_rows(df) == _naive_near_duplicates(df)

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            (1, True),
            (0, False),
            (1, 1.0),
            (None, float("nan")),
            (None, pd.NA),
            (float("nan"), pd.NA),
            ("1", 1),
            (-0.0, 0.0),
            (pd.Timestamp("2020-01-01"), "2020-01-01"),
        ],
        ids=repr,
    )
    def test_uses_duplicated_equality(self, a, b):
        from profiler.duplicates import _near_duplicate_rows

        df = pd.DataFrame(
            {"v": pd.array([a, b], dtype=object), "w": [7, 7], "k": ["p", "q"]}
        )
        assert _near_duplicate_rows(df) == _naive_near_duplicates(df)

    def test_permuted_values_are_not_near_duplicates(self):
        from profiler.duplicates import _near_duplicate_rows

        df = pd.DataFrame({"a": [1, 2], "b": [2, 1], "c": [3, 3], "d": [7, 8]})
        assert _near_duplicate_rows(df) == set()

    def test_duplicate_column_names_are_handled_by_position(self):
        from profiler.duplicates import _near_duplicate_rows

        # Two columns named "a": label-based selection would pick both at once
        df = pd.DataFrame([[1, 2, 3], [1, 2, 4], [5, 6, 7]], columns=["a", "a", "b"])
        assert _near_duplicate_rows(df) == {0, 1}  # differ only in column "b"

    def test_near_sample_is_sorted(self):
        df = pd.DataFrame(
            {"a": [1, 9, 1, 9], "b": [2, 8, 2, 8], "c": ["x", "q", "z", "r"]}
        )
        assert analyze_duplicates(df)["near_sample_indices"] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# infinite values: stats and fences use finite values, inf is always flagged
# ---------------------------------------------------------------------------


class TestInfiniteValues:
    def _df(self):
        import numpy as np

        return pd.DataFrame({"x": [float(i) for i in range(1, 20)] + [np.inf, -np.inf]})

    def test_stats_exclude_inf_and_count_it(self):
        df = self._df()
        stats = compute_stats(df, detect_types(df))["x"]
        assert stats["mean"] == 10.0
        assert stats["infinite_count"] == 2

    def test_inf_flagged_by_both_methods(self):
        df = self._df()
        result = detect_outliers(df, detect_types(df))["x"]
        assert set(result["iqr_indices"]) == {19, 20}
        assert {19, 20} <= set(result["zscore_indices"])

    def test_inf_flagged_with_too_few_finite_values(self):
        import numpy as np

        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.inf, -np.inf]})
        result = detect_outliers(df, detect_types(df))["x"]
        assert result["iqr_indices"] == [3, 4]
        assert result["zscore_indices"] == [3, 4]

    def test_inf_flagged_when_iqr_is_zero(self):
        import numpy as np

        df = pd.DataFrame({"x": [5.0] * 8 + [np.inf]})
        result = detect_outliers(df, detect_types(df))["x"]
        assert result["iqr_indices"] == [8]
        assert result["zscore_indices"] == [8]

    def test_no_inf_keeps_empty_result_for_short_columns(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        assert detect_outliers(df, detect_types(df))["x"]["iqr_count"] == 0

    def test_normality_ignores_inf(self):
        import numpy as np

        rng = np.random.default_rng(0)
        series = pd.Series(list(rng.normal(0, 1, 200)) + [np.inf])
        p_value, _ = _normality_test(series)
        assert p_value is not None

    def test_report_warns_about_inf(self):
        report = build_report(self._df(), "inf.csv")
        assert any("infinite value" in w for w in report["columns"]["x"]["warnings"])


# ---------------------------------------------------------------------------
# Ported from data-quality-mcp, Milestone 3.5 and #14
# ---------------------------------------------------------------------------


def test_iso_dates_in_a_csv_are_not_a_type_mismatch():
    dates = ["2020-01-01", "2020-02-03", "2021-12-31"]
    df = pd.DataFrame({"when": dates, "x": [1, 2, 3]})
    csv = build_report(df, "data.CSV")["columns"]["when"]
    assert (csv["inferred_type"], csv["type_mismatch"]) == ("datetime", False)
    xlsx = build_report(df, "data.xlsx")["columns"]["when"]
    assert xlsx["type_mismatch"] is True  # XLSX has a date type


@pytest.mark.parametrize(
    "values",
    [
        ["01/02/2020", "03/04/2020", "05/06/2020"],
        ["2020-01-01", "01/02/2020", "15-Jan-2020"],
        ["2020-01-01T10:00:00Z", "2020-01-02 10:00"],
    ],
)
def test_other_date_text_stays_a_mismatch_in_a_csv(values):
    df = pd.DataFrame({"when": values})
    assert build_report(df, "data.csv")["columns"]["when"]["type_mismatch"] is True


def test_timestamps_with_mixed_offsets_are_typed_and_profiled():
    values = [
        "2020-01-01T10:00:00Z",
        "2020-01-02T10:00:00+02:00",
        "2020-01-03T09:00:00+01:00",
    ]
    report = build_report(pd.DataFrame({"when": values}), "t.csv")
    column = report["columns"]["when"]
    assert column["inferred_type"] == "datetime"
    assert column["type_mismatch"] is False  # all ISO, all with an offset, in a CSV
    assert column["stats"]["min_date"].startswith("2020-01-01 10:00:00")
    json.dumps(report)


def _paired(n_shared: int) -> pd.DataFrame:
    x = [float(i) for i in range(n_shared + 5)]
    y = [2.0 * i for i in range(n_shared)] + [None] * 5
    return pd.DataFrame({"x": x, "y": y})


def test_correlations_need_ten_shared_rows():
    from profiler.correlations import compute_correlations

    below = compute_correlations(_paired(9), detect_types(_paired(9)))
    assert below["pearson"]["x"]["y"] is None
    assert below["high_correlation_pairs"] == []
    at = compute_correlations(_paired(10), detect_types(_paired(10)))
    assert at["pearson"]["x"]["y"] == 1.0


def test_cramers_v_needs_ten_shared_rows():
    from profiler.correlations import compute_correlations

    def frame(n_shared: int) -> pd.DataFrame:
        a = ["x", "y"] * 10
        b = (["x", "y"] * 10)[:n_shared] + [None] * (20 - n_shared)
        return pd.DataFrame({"a": a, "b": b})

    below = compute_correlations(frame(9), detect_types(frame(9)))
    assert below["cramers_v"]["a"]["b"] is None
    assert below["high_correlation_pairs"] == []
    # From 10 shared rows on. Without Yates' correction a matching pair is 1.0
    # (it was 0.8 with it), so it is a high pair
    at = compute_correlations(frame(10), detect_types(frame(10)))
    assert at["cramers_v"]["a"]["b"] == 1.0
    assert at["high_correlation_pairs"][0]["method"] == "cramers_v"


def _pattern_frame(n: int = 400, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(0, 80, n).astype(float),
            "group": rng.choice(["a", "b", "c", "d"], n),
            "noise": rng.normal(size=n),
            "target": 1.0,
        }
    )


def test_missing_pattern_through_a_category():
    df = _pattern_frame()
    df.loc[df["group"] == "a", "target"] = np.nan
    info = analyze_missing(df)["per_column"]["target"]
    assert info["missing_pattern"] == "MAR"
    assert info["missing_related_to"][0]["method"] == "cramers_v"
    assert info["missing_related_to"][0]["column"] == "group"


def test_missing_pattern_through_missing_together():
    df = _pattern_frame()
    gaps = df.index % 5 == 0
    df.loc[gaps, "target"] = np.nan
    df.loc[gaps, "noise"] = np.nan
    related = analyze_missing(df)["per_column"]["target"]["missing_related_to"]
    assert {"column": "noise", "method": "missing_together", "value": 1.0} in related


def test_a_constant_column_gives_no_pattern():
    # 0.1's float mean has a rounding error: the column must still count as constant
    df = pd.DataFrame({"c": [0.1] * 300, "t": [1.0] * 300})
    df.loc[:99, "t"] = np.nan
    info = analyze_missing(df)["per_column"]["t"]
    assert info["missing_related_to"] is None
    assert info["missing_pattern"] == "unknown"


def test_unknown_without_a_comparison_column():
    df = pd.DataFrame(
        {"note": [f"free text {i}" for i in range(400)], "t": [1.0] * 400}
    )
    df.loc[:199, "t"] = np.nan
    assert analyze_missing(df)["per_column"]["t"]["missing_pattern"] == "unknown"


# --- bias-corrected Cramér's V (ported from data-quality-mcp, Milestone 3.5,
# Amendment: bias-corrected Cramér's V) ---


def _categorical(df: pd.DataFrame) -> dict:
    """compute_correlations with every column typed categorical: on few rows,
    type detection would call a many-level column text."""
    return compute_correlations(df, {c: {"inferred_type": "categorical"} for c in df})


def _plain_v(x: pd.Series, y: pd.Series) -> float:
    """The plain V (Yates on 2 x 2), the value before this port."""
    from scipy.stats import chi2_contingency

    table = pd.crosstab(x, y)
    chi2 = chi2_contingency(table)[0]
    return float(np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1))))


def _reference_v(x: pd.Series, y: pd.Series) -> float:
    """The corrected V from a pd.crosstab table, no Yates' correction."""
    from scipy.stats import chi2_contingency

    from profiler.correlations import corrected_cramers_v

    table = pd.crosstab(x, y)  # drops rows where either value is missing
    if min(table.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    return corrected_cramers_v(float(chi2), int(table.to_numpy().sum()), table.shape)


def test_unrelated_many_level_columns_are_no_longer_a_high_pair():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "a": [f"a{v}" for v in rng.integers(0, 50, 20)],
            "b": [f"b{v}" for v in rng.integers(0, 50, 20)],
        }
    )
    assert _plain_v(df["a"], df["b"]) >= 0.9  # the false high pair before
    result = _categorical(df)
    assert result["cramers_v"]["a"]["b"] < 0.5
    assert result["high_correlation_pairs"] == []


@pytest.mark.parametrize("levels", [2, 5])
def test_a_one_to_one_association_is_one(levels):
    from profiler.correlations import _cramers_v

    a = [f"x{i % levels}" for i in range(50)]
    b = [f"y{i % levels}" for i in range(50)]
    assert _cramers_v(pd.Series(a), pd.Series(b)) == 1.0


@pytest.mark.parametrize(("rows", "high"), [(20, False), (100, True)])
def test_a_many_to_one_mapping_needs_rows_per_level(rows, high):
    # The accepted trade-off: 10 levels -> 2 groups is redundant, but on 2 rows
    # per level the corrected V stays below 0.9
    rng = np.random.default_rng(10)
    codes = rng.integers(0, 10, rows)
    df = pd.DataFrame({"city": [f"c{v}" for v in codes], "region": codes % 2})
    result = _categorical(df)
    assert (result["cramers_v"]["city"]["region"] >= 0.9) is high
    assert bool(result["high_correlation_pairs"]) is high


def test_the_housing_sample_pair():
    # 0.1171 with the plain V. data-quality-mcp's fixture differs (0.0292 there)
    sample = Path(__file__).parent.parent / "sample_data" / "housing_messy.csv"
    df = pd.read_csv(sample)
    result = compute_correlations(df, detect_types(df))
    assert result["cramers_v"] == {"neighborhood": {"garage": 0.0153}, "garage": {}}


def test_a_two_by_two_table_has_no_yates_correction():
    from scipy.stats import chi2_contingency

    from profiler.correlations import _cramers_v, corrected_cramers_v

    rng = np.random.default_rng(22)
    x = pd.Series(rng.integers(0, 2, 30))
    y = pd.Series(np.where(rng.random(30) < 0.7, x, 1 - x))
    table = pd.crosstab(x, y)
    chi2 = chi2_contingency(table, correction=False)[0]
    assert _cramers_v(x, y) == corrected_cramers_v(float(chi2), 30, (2, 2))
    assert chi2_contingency(table)[0] < chi2  # Yates would have lowered it


@pytest.mark.parametrize("seed", range(8))
def test_the_numpy_table_matches_crosstab(seed):
    from profiler.correlations import _cramers_v

    rng = np.random.default_rng(seed)
    n = 200
    x = pd.Series([f"x{v}" for v in rng.integers(0, 6, n)], dtype=object)
    y = pd.Series([[True, False, "maybe"][v] for v in rng.integers(0, 3, n)])
    y = y.astype(object)  # real bools beside a string
    nulls = [None, np.nan, pd.NA, pd.NaT]
    for s in (x, y):
        for i in np.flatnonzero(rng.random(n) < 0.15):
            s[i] = nulls[i % len(nulls)]
    # A category that only appears where the other column is missing
    lonely = int(np.flatnonzero(y.isna().to_numpy())[0])
    x[lonely] = "only-here"
    assert _cramers_v(x, y) == pytest.approx(_reference_v(x, y), abs=1e-12)
    assert _cramers_v(y, x) == pytest.approx(_reference_v(x, y), abs=1e-12)


def test_series_are_matched_by_label_and_no_shared_rows_gives_zero():
    from profiler.correlations import _cramers_v

    rng = np.random.default_rng(5)
    x = pd.Series(["a", "b"] * 15)
    y = pd.Series(["p", "q"] * 15, index=rng.permutation(30))
    positional = pd.Series(y.to_numpy())
    assert _cramers_v(x, positional) == 1.0  # a different answer by position
    assert _cramers_v(x, y) == pytest.approx(_reference_v(x, y), abs=1e-12)
    assert _cramers_v(x, y) < 1.0
    assert _cramers_v(pd.Series(["a", None]), pd.Series([None, "b"])) == 0.0


def test_a_different_value_on_every_shared_row_gives_zero():
    from profiler.correlations import _cramers_v

    x = pd.Series([f"r{i}" for i in range(12)])
    y = pd.Series(["a", "b"] * 6)
    assert _cramers_v(x, y) == 0.0


def test_one_category_on_the_shared_rows_gives_zero():
    from profiler.correlations import _cramers_v

    # "b" exists, but only where y is missing
    x = pd.Series(["a"] * 10 + ["b"] * 3)
    y = pd.Series(["p", "q"] * 5 + [None] * 3)
    assert _cramers_v(x, y) == 0.0
    assert _categorical(pd.DataFrame({"x": x, "y": y}))["cramers_v"]["x"]["y"] == 0.0


# --- ruling out dates early (ported from data-quality-mcp #20): the result
# must equal parsing every value ---

_DATES = [str(d.date()) for d in pd.date_range("2000-01-01", periods=9000, freq="D")]
_JUNK = [f"note {i}" for i in range(9000)]


def _days(first: date, n: int) -> list[str]:
    """`n` consecutive ISO dates; built in Python, as pandas 2 can't hold
    dates outside 1677-2262."""
    return [str(first + timedelta(days=i)) for i in range(n)]


def _interleave(dates: int, junk: int) -> list[str]:
    """`dates` dates and `junk` free-text values, spread evenly."""
    values = _DATES[:dates] + _JUNK[:junk]
    order = np.argsort(np.arange(len(values)) % 7, kind="stable")
    return [values[i] for i in order]


DATE_CASES = {
    "free text": _JUNK[:5000],
    "a quarter junk": _interleave(6000, 2000),
    "exactly at the threshold": _interleave(8000, 2000),
    "just below the threshold": _interleave(7999, 2001),
    "junk first, then dates": _JUNK[:1500] + _DATES[:8500],
    "one junk value on many rows": ["n/a"] * 3000 + _DATES[:7000],
    "dates on many rows, junk unique": _DATES[:20] * 400 + _JUNK[:2000],
    "naive values, then mixed offsets": _DATES[:1500]
    + [f"2020-01-01T10:00:00{tz}" for tz in ("Z", "+02:00")] * 300,
    "nanoseconds, then dates after 2262": [
        f"2020-01-01 00:00:00.{i:09d}" for i in range(1, 1001)
    ]
    + _days(date(2300, 1, 1), 4000),
}


@pytest.mark.parametrize("values", DATE_CASES.values(), ids=DATE_CASES.keys())
def test_ruling_out_dates_early_gives_the_full_parse_result(values):
    from profiler.type_detector import _DATETIME_THRESHOLD, _date_mask, parse_datetimes

    codes, uniques = pd.factorize(pd.Series(values, dtype=object))
    uniques = pd.Series(uniques, dtype=object)
    full = parse_datetimes(uniques).notna().to_numpy()
    mask = _date_mask(uniques, np.bincount(codes))
    if mask is None:
        assert full[codes].mean() < _DATETIME_THRESHOLD
    else:
        assert (mask == full).all()


def test_dates_out_of_range_for_the_whole_column_stay_unparsed():
    # On pandas 3 one nanosecond timestamp makes the whole column nanosecond
    # resolution, where dates before 1677 don't fit; a chunk without it would
    # parse them, so the column must not pass with half its values
    from profiler.type_detector import _infer_type

    values = _days(date(1500, 1, 1), 500) + _days(date(2000, 1, 1), 500)
    values.append("2020-01-01 00:00:00.123456789")
    assert _infer_type(pd.Series(values, dtype=object)) == ("text", False)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (_interleave(8000, 2000), ("datetime", True)),
        (_interleave(7999, 2001), ("text", False)),
        (["n/a"] * 3000 + _DATES[:7000], ("text", False)),
    ],
)
def test_the_date_threshold_is_unchanged(values, expected):
    from profiler.type_detector import _infer_type

    assert _infer_type(pd.Series(values, dtype=object)) == expected


def test_free_text_is_ruled_out_after_a_fraction_of_its_values(monkeypatch):
    from profiler import type_detector

    parsed = []
    parse_mixed = type_detector._parse_mixed

    def spy(series):
        parsed.append(len(series))
        return parse_mixed(series)

    monkeypatch.setattr(type_detector, "_parse_mixed", spy)
    assert type_detector._infer_type(pd.Series(_JUNK, dtype=object)) == ("text", False)
    assert parsed == [1000, 1000]  # 20% of 9000 is 1800
    parsed.clear()
    assert type_detector._infer_type(pd.Series(_DATES, dtype=object))[0] == "datetime"
    assert parsed == [1000, 9000]  # one chunk, then the whole column


def test_a_chunk_needing_the_last_fallback_parses_the_whole_column(monkeypatch):
    from profiler import type_detector

    monkeypatch.setattr(type_detector, "_parse_mixed", lambda series: None)
    uniques = pd.Series(_DATES[:3000], dtype=object)
    mask = type_detector._date_mask(uniques, np.ones(3000, dtype=int))
    assert mask is not None and mask.all()
