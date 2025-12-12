from pathlib import Path

import numpy as np
import pandas as pd


def get_abs_path(rootdir, filename):
    return Path(rootdir / filename).resolve()


def to_csv(outputfile: Path, column_labels: tuple, *column_values) -> None:
    if len(column_labels) != len(column_values):
        raise ValueError(
            "Number of column labels must match number of column value arrays."
        )

    df = pd.DataFrame(np.column_stack(column_values), columns=column_labels)
    df.to_csv(outputfile, header=True, index=False, float_format="%.3f")


def to_excel(
    writer: pd.ExcelWriter, column_labels: tuple, sheet_name: str, *column_values
) -> None:
    if len(column_labels) != len(column_values):
        raise ValueError(
            "Number of column labels must match number of column value arrays."
        )

    df = pd.DataFrame(np.column_stack(column_values), columns=column_labels)
    df.to_excel(
        writer, sheet_name=sheet_name, header=True, index=False, float_format="%.3f"
    )
