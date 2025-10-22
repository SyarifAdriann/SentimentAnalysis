"""Data validation helpers for Tweets dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass(slots=True)
class DataQualitySnapshot:
    row_count: int
    column_count: int
    duplicate_rows: int
    missing_values: Dict[str, int]
    numeric_summary: Dict[str, Dict[str, float]]

    def to_markdown(self) -> str:
        """Render the snapshot as markdown for reporting."""

        lines = ["# Data Quality Report", ""]
        lines.append(f"- Rows: {self.row_count}")
        lines.append(f"- Columns: {self.column_count}")
        lines.append(f"- Duplicate rows: {self.duplicate_rows}")
        lines.append("")
        lines.append("## Missing Values")
        for column, count in self.missing_values.items():
            lines.append(f"- {column}: {count}")
        lines.append("")
        if self.numeric_summary:
            lines.append("## Numeric Column Summary")
            for column, summary in self.numeric_summary.items():
                lines.append(f"- {column}:")
                for key, value in summary.items():
                    lines.append(f"  - {key}: {value:.4f}")
        return "\n".join(lines)


NUMERIC_STATISTICS = ("mean", "std", "min", "max")


def build_snapshot(df: pd.DataFrame) -> DataQualitySnapshot:
    missing = df.isna().sum().to_dict()
    duplicates = df.duplicated().sum()

    numeric_cols = df.select_dtypes(include=["number"]).columns
    numeric_summary: Dict[str, Dict[str, float]] = {}
    if len(numeric_cols) > 0:
        describe = df[numeric_cols].describe().transpose()
        for column in numeric_cols:
            stats = {}
            for key in NUMERIC_STATISTICS:
                stats[key] = float(describe.loc[column, key])
            numeric_summary[column] = stats

    snapshot = DataQualitySnapshot(
        row_count=int(len(df)),
        column_count=int(len(df.columns)),
        duplicate_rows=int(duplicates),
        missing_values={k: int(v) for k, v in missing.items()},
        numeric_summary=numeric_summary,
    )
    return snapshot


def compare_column_set(
    df: pd.DataFrame,
    expected_columns: Optional[list[str]] = None,
) -> Dict[str, list[str]]:
    """Return missing and unexpected column names."""

    if expected_columns is None:
        expected_columns = []

    actual_columns = list(df.columns)

    missing = [col for col in expected_columns if col not in actual_columns]
    unexpected = [col for col in actual_columns if col not in expected_columns]
    return {"missing": missing, "unexpected": unexpected}
