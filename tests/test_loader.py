import io

import pandas as pd
import pytest

from profiler.loader import load_file


class UploadedFileMock:
    def __init__(self, content: bytes, name: str):
        self._buf = io.BytesIO(content)
        self.name = name

    def read(self):
        return self._buf.read()

    def seek(self, pos):
        self._buf.seek(pos)


def test_csv_comma_separated():
    f = UploadedFileMock(b"col1,col2\n1,a\n2,b\n", "data.csv")
    df = load_file(f)
    assert list(df.columns) == ["col1", "col2"]
    assert len(df) == 2


def test_csv_semicolon_separated():
    f = UploadedFileMock(b"col1;col2\n1;a\n2;b\n", "data.csv")
    df = load_file(f)
    assert list(df.columns) == ["col1", "col2"]
    assert len(df) == 2


def test_csv_returns_dataframe():
    f = UploadedFileMock(b"x,y\n10,20\n", "test.csv")
    assert isinstance(load_file(f), pd.DataFrame)


def test_unsupported_extension_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        load_file(UploadedFileMock(b"", "data.json"))
