MAX_UPLOAD_SIZE_MB = 200
MISSING_THRESHOLD = 0.30              # Bar chart: columns above this % are flagged red
COLUMN_WARNING_MISSING_THRESHOLD = 0.50  # Column warning: raise a warning when missing % exceeds this
OUTLIER_IQR_MULTIPLIER = 1.5
OUTLIER_ZSCORE_THRESHOLD = 3.0
HIGH_CORRELATION_THRESHOLD = 0.9
HIGH_CARDINALITY_THRESHOLD = 0.5
MAX_PREVIEW_ROWS = 10
QUALITY_GRADE_THRESHOLDS = {"A": 90, "B": 80, "C": 70, "D": 60}

# Ported from data-quality-mcp, Milestone 3.5 (planning/milestone-3.5.md there)
MIN_CORRELATION_ROWS = 10             # Rows with both values a correlation needs
# Missing-value pattern: which observed columns go with a column's missing values
MAX_MISSING_PATTERN_COLUMNS = 50      # Columns with missing values that are tested
MIN_MISSING_PATTERN_ROWS = 10         # Missing and present rows each tested pair needs
MISSING_ASSOCIATION_THRESHOLD = 0.3   # Minimum |r|, Cramér's V or phi to count
MISSING_ASSOCIATION_MAX_P = 0.001     # Divided by the pairs tested (Bonferroni)
MAX_MISSING_ASSOCIATIONS = 5          # Listed per column, strongest first
MAX_PATTERN_NUMERIC_COLUMNS = 50      # Comparison columns for the pattern test...
MAX_PATTERN_CATEGORICAL_COLUMNS = 15  # ...categorical ones...
MAX_PATTERN_CARDINALITY = 50          # ...with at most this many distinct values
