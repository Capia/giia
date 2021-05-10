from typing import Optional

import pandas as pd
import pydantic
from gluonts.dataset.split.splitter import AbstractBaseSplitter, TimeSeriesSlice

"""
Duplicate of gluonts' DateSplitter with one small bugfix. Instead of using `unit` in Timedelta, we instead use `freq`
"""


class DateSplitter(AbstractBaseSplitter, pydantic.BaseModel):
    prediction_length: int
    split_date: pd.Timestamp
    max_history: Optional[int] = None

    def _train_slice(self, item: TimeSeriesSlice) -> TimeSeriesSlice:
        # the train-slice includes everything up to (including) the split date
        return item[: self.split_date]

    def _test_slice(
            self, item: TimeSeriesSlice, offset: int = 0
    ) -> TimeSeriesSlice:
        freq = item.start.freqstr
        return item[: self.split_date + pd.Timedelta(self.prediction_length + offset, freq="1min")]
