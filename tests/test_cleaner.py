import numpy as np
import pandas as pd
import pytest

from app.components.cleaning_panel import _step_priority
from cleaner.duplicate_handler import remove_duplicates
from cleaner.missing_handler import handle_missing
from cleaner.outlier_handler import handle_outliers
from cleaner.pipeline import CleaningPipeline
from cleaner.type_fixer import fix_types


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_with_missing():
    return pd.DataFrame({
        "x": [1.0, 2.0, None, 4.0, None],
        "cat": ["a", "b", None, "a", "b"],
    })


@pytest.fixture
def df_with_dups():
    return pd.DataFrame({
        "a": [1, 2, 1, 3],
        "b": ["x", "y", "x", "z"],
    })


@pytest.fixture
def df_mixed_types():
    return pd.DataFrame({
        "price_str": ["1,200", "$500", "300", "nan", "1500"],
        "date_str": ["2020-01-01", "2021-06-15", "2022-03-20", "2023-08-10", "2019-11-05"],
        "flag_str": ["yes", "no", "yes", "no", "yes"],
    })


@pytest.fixture
def df_with_outliers():
    vals = list(range(1, 21)) + [1000]
    return pd.DataFrame({"x": vals})


# ---------------------------------------------------------------------------
# handle_missing
# ---------------------------------------------------------------------------


class TestHandleMissing:
    def test_drop_rows(self, df_with_missing):
        out, log = handle_missing(df_with_missing, "drop_rows", "x")
        assert out["x"].isnull().sum() == 0
        assert len(out) == 3
        assert log["rows_removed"] == 2

    def test_drop_column(self, df_with_missing):
        out, log = handle_missing(df_with_missing, "drop_column", "x")
        assert "x" not in out.columns
        assert "cat" in out.columns

    def test_fill_mean(self, df_with_missing):
        out, log = handle_missing(df_with_missing, "fill_mean", "x")
        assert out["x"].isnull().sum() == 0
        assert log["missing_after"] == 0

    def test_fill_median(self, df_with_missing):
        out, log = handle_missing(df_with_missing, "fill_median", "x")
        assert out["x"].isnull().sum() == 0

    def test_fill_mode(self, df_with_missing):
        out, log = handle_missing(df_with_missing, "fill_mode", "cat")
        assert out["cat"].isnull().sum() == 0

    def test_fill_constant(self, df_with_missing):
        out, log = handle_missing(df_with_missing, "fill_constant", "x", value=99.0)
        assert (out["x"] == 99.0).sum() == 2
        assert out["x"].isnull().sum() == 0

    def test_ffill(self, df_with_missing):
        out, log = handle_missing(df_with_missing, "ffill", "x")
        assert out["x"].iloc[2] == 2.0  # propagated from previous row

    def test_bfill(self, df_with_missing):
        out, log = handle_missing(df_with_missing, "bfill", "x")
        assert out["x"].iloc[2] == 4.0  # propagated from next row

    def test_non_destructive(self, df_with_missing):
        original_nulls = df_with_missing["x"].isnull().sum()
        handle_missing(df_with_missing, "fill_mean", "x")
        assert df_with_missing["x"].isnull().sum() == original_nulls

    def test_unknown_strategy_raises(self, df_with_missing):
        with pytest.raises(ValueError, match="Unknown strategy"):
            handle_missing(df_with_missing, "magic_fix", "x")

    def test_log_schema(self, df_with_missing):
        _, log = handle_missing(df_with_missing, "fill_mean", "x")
        for key in ("action", "column", "strategy", "rows_before", "rows_after",
                    "rows_removed", "missing_before", "missing_after"):
            assert key in log


# ---------------------------------------------------------------------------
# remove_duplicates
# ---------------------------------------------------------------------------


class TestRemoveDuplicates:
    def test_keep_first(self, df_with_dups):
        out, log = remove_duplicates(df_with_dups, keep="first")
        assert len(out) == 3
        assert log["rows_removed"] == 1

    def test_keep_last(self, df_with_dups):
        out, log = remove_duplicates(df_with_dups, keep="last")
        assert len(out) == 3

    def test_remove_all(self, df_with_dups):
        out, log = remove_duplicates(df_with_dups, keep="none")
        # Both copies of row (1, "x") are removed
        assert len(out) == 2
        assert log["rows_removed"] == 2

    def test_no_duplicates(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        out, log = remove_duplicates(df)
        assert len(out) == 3
        assert log["rows_removed"] == 0

    def test_non_destructive(self, df_with_dups):
        original_len = len(df_with_dups)
        remove_duplicates(df_with_dups, keep="first")
        assert len(df_with_dups) == original_len

    def test_log_schema(self, df_with_dups):
        _, log = remove_duplicates(df_with_dups)
        for key in ("action", "column", "strategy", "rows_before", "rows_after", "rows_removed"):
            assert key in log


# ---------------------------------------------------------------------------
# fix_types
# ---------------------------------------------------------------------------


class TestFixTypes:
    def test_to_numeric_strips_commas(self, df_mixed_types):
        out, log = fix_types(df_mixed_types, "price_str", "numeric")
        assert out["price_str"].dtype in (float, np.float64)
        assert out["price_str"].iloc[0] == pytest.approx(1200.0)

    def test_to_numeric_strips_currency(self, df_mixed_types):
        out, _ = fix_types(df_mixed_types, "price_str", "numeric")
        assert out["price_str"].iloc[1] == pytest.approx(500.0)

    def test_to_datetime(self, df_mixed_types):
        out, log = fix_types(df_mixed_types, "date_str", "datetime")
        assert pd.api.types.is_datetime64_any_dtype(out["date_str"])

    def test_to_categorical(self):
        df = pd.DataFrame({"cat": ["a", "b", "a", "c"]})
        out, log = fix_types(df, "cat", "categorical")
        assert hasattr(out["cat"], "cat")

    def test_to_string_preserves_nan(self):
        df = pd.DataFrame({"x": [1.0, None, 3.0]})
        out, log = fix_types(df, "x", "string")
        assert out["x"].isnull().sum() == 1

    def test_to_boolean(self, df_mixed_types):
        out, log = fix_types(df_mixed_types, "flag_str", "boolean")
        assert set(out["flag_str"].dropna().unique()).issubset({True, False})

    def test_unknown_type_raises(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="Unknown target_type"):
            fix_types(df, "x", "imaginary_type")

    def test_non_destructive(self, df_mixed_types):
        original_dtype = df_mixed_types["price_str"].dtype
        fix_types(df_mixed_types, "price_str", "numeric")
        assert df_mixed_types["price_str"].dtype == original_dtype

    def test_log_schema(self, df_mixed_types):
        _, log = fix_types(df_mixed_types, "price_str", "numeric")
        for key in ("action", "column", "strategy", "original_dtype", "new_dtype"):
            assert key in log


# ---------------------------------------------------------------------------
# handle_outliers
# ---------------------------------------------------------------------------


class TestHandleOutliers:
    def test_remove_strategy(self, df_with_outliers):
        indices = [20]  # the outlier row (value 1000)
        out, log = handle_outliers(df_with_outliers, "x", indices, "remove")
        assert 1000 not in out["x"].values
        assert log["rows_removed"] == 1

    def test_cap_iqr_strategy(self, df_with_outliers):
        indices = [20]
        out, log = handle_outliers(df_with_outliers, "x", indices, "cap_iqr")
        assert len(out) == len(df_with_outliers)  # no rows removed
        assert out["x"].max() < 1000  # value was capped

    def test_replace_nan_strategy(self, df_with_outliers):
        indices = [20]
        out, log = handle_outliers(df_with_outliers, "x", indices, "replace_nan")
        assert len(out) == len(df_with_outliers)
        assert pd.isna(out.loc[20, "x"])

    def test_invalid_indices_ignored(self, df_with_outliers):
        out, log = handle_outliers(df_with_outliers, "x", [999], "remove")
        assert len(out) == len(df_with_outliers)
        assert log["outliers_treated"] == 0

    def test_unknown_strategy_raises(self, df_with_outliers):
        with pytest.raises(ValueError, match="Unknown strategy"):
            handle_outliers(df_with_outliers, "x", [20], "teleport")

    def test_non_destructive(self, df_with_outliers):
        original_max = df_with_outliers["x"].max()
        handle_outliers(df_with_outliers, "x", [20], "cap_iqr")
        assert df_with_outliers["x"].max() == original_max

    def test_log_schema(self, df_with_outliers):
        _, log = handle_outliers(df_with_outliers, "x", [20], "remove")
        for key in ("action", "column", "strategy", "outliers_treated",
                    "rows_before", "rows_after", "rows_removed"):
            assert key in log


# ---------------------------------------------------------------------------
# CleaningPipeline
# ---------------------------------------------------------------------------


class TestCleaningPipeline:
    def test_add_and_count_steps(self):
        p = CleaningPipeline()
        p.add_step({"action": "remove_duplicates", "keep": "first"})
        assert len(p) == 1

    def test_execute_applies_steps(self, df_with_missing):
        p = CleaningPipeline()
        p.add_step({"action": "handle_missing", "column": "x", "strategy": "fill_mean"})
        out = p.execute(df_with_missing)
        assert out["x"].isnull().sum() == 0

    def test_execute_does_not_modify_original(self, df_with_missing):
        p = CleaningPipeline()
        p.add_step({"action": "handle_missing", "column": "x", "strategy": "fill_mean"})
        p.execute(df_with_missing)
        assert df_with_missing["x"].isnull().sum() == 2  # unchanged

    def test_log_populated_after_execute(self, df_with_missing):
        p = CleaningPipeline()
        p.add_step({"action": "handle_missing", "column": "x", "strategy": "fill_median"})
        p.execute(df_with_missing)
        assert len(p.get_log()) == 1
        assert p.get_log()[0]["action"] == "handle_missing"

    def test_multi_step_pipeline(self, df_with_missing, df_with_dups):
        combined = pd.concat([df_with_missing, df_with_missing], ignore_index=True)
        p = CleaningPipeline()
        p.add_step({"action": "handle_missing", "column": "x", "strategy": "fill_mean"})
        p.add_step({"action": "remove_duplicates", "keep": "first"})
        out = p.execute(combined)
        assert out["x"].isnull().sum() == 0
        assert out.duplicated().sum() == 0

    def test_clear_resets_steps_and_log(self, df_with_missing):
        p = CleaningPipeline()
        p.add_step({"action": "handle_missing", "column": "x", "strategy": "fill_mean"})
        p.execute(df_with_missing)
        p.clear()
        assert len(p) == 0
        assert len(p.get_log()) == 0

    def test_empty_pipeline_returns_copy(self, df_with_missing):
        p = CleaningPipeline()
        out = p.execute(df_with_missing)
        pd.testing.assert_frame_equal(out, df_with_missing)

    def test_unknown_action_raises(self, df_with_missing):
        p = CleaningPipeline()
        p.add_step({"action": "explode_data"})
        with pytest.raises(ValueError, match="Unknown action"):
            p.execute(df_with_missing)

    def test_fix_type_step(self, df_mixed_types):
        p = CleaningPipeline()
        p.add_step({"action": "fix_type", "column": "price_str", "target_type": "numeric"})
        out = p.execute(df_mixed_types)
        assert pd.api.types.is_numeric_dtype(out["price_str"])

    def test_outlier_step(self, df_with_outliers):
        p = CleaningPipeline()
        p.add_step({
            "action": "handle_outliers",
            "column": "x",
            "indices": [20],
            "strategy": "remove",
        })
        out = p.execute(df_with_outliers)
        assert 1000 not in out["x"].values


# ---------------------------------------------------------------------------
# _step_priority (cleaning_panel)
# ---------------------------------------------------------------------------


class TestStepPriority:
    """Verify the three-tier ordering that keeps profiler indices valid.

    Priority 0 — no row change (type fixes, fills, cap, replace_nan)
    Priority 1 — outlier remove (uses original row indices from profiler)
    Priority 2 — row-dropping ops (drop_rows, drop_column, remove_duplicates)
    """

    def test_fix_type_is_priority_zero(self):
        assert _step_priority({"action": "fix_type", "column": "x", "target_type": "numeric"}) == 0

    def test_fill_mean_is_priority_zero(self):
        assert _step_priority({"action": "handle_missing", "column": "x", "strategy": "fill_mean"}) == 0

    def test_fill_constant_is_priority_zero(self):
        assert _step_priority({"action": "handle_missing", "column": "x", "strategy": "fill_constant"}) == 0

    def test_cap_iqr_is_priority_zero(self):
        assert _step_priority({"action": "handle_outliers", "column": "x", "strategy": "cap_iqr"}) == 0

    def test_replace_nan_is_priority_zero(self):
        assert _step_priority({"action": "handle_outliers", "column": "x", "strategy": "replace_nan"}) == 0

    def test_outlier_remove_is_priority_one(self):
        assert _step_priority({"action": "handle_outliers", "column": "x", "strategy": "remove"}) == 1

    def test_drop_rows_is_priority_two(self):
        assert _step_priority({"action": "handle_missing", "column": "x", "strategy": "drop_rows"}) == 2

    def test_drop_column_is_priority_two(self):
        assert _step_priority({"action": "handle_missing", "column": "x", "strategy": "drop_column"}) == 2

    def test_remove_duplicates_is_priority_two(self):
        assert _step_priority({"action": "remove_duplicates"}) == 2

    def test_sort_order_is_correct(self):
        steps = [
            {"action": "handle_missing", "column": "x", "strategy": "drop_rows"},
            {"action": "handle_outliers", "column": "x", "strategy": "remove"},
            {"action": "fix_type", "column": "x", "target_type": "numeric"},
            {"action": "remove_duplicates"},
            {"action": "handle_missing", "column": "y", "strategy": "fill_median"},
        ]
        sorted_steps = sorted(steps, key=_step_priority)
        priorities = [_step_priority(s) for s in sorted_steps]
        assert priorities == sorted(priorities)
