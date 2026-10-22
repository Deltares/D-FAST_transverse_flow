import numpy as np

from dfasttf.batch.plotting.longitudinal_flow import LongitudinalFlow


def test_create_tide_timeseries_figure_with_annotations(tmp_path):
    filename = tmp_path / "longitudinal-flow-tide.png"
    time = np.array(
        ["2024-01-01T00", "2024-01-01T01", "2024-01-01T02"],
        dtype="datetime64[h]",
    )
    upar = np.array([[-0.5, 0.2], [0.3, -0.1], [0.8, 0.4]])

    LongitudinalFlow().create_figure_tide_timeseries(
        time=time,
        upar_tn=upar,
        rkm=np.array([1000.0, 2000.0]),
        filename=filename,
        threshold=1.0,
        annotation="test annotation",
        idx_ebb=np.array([0, 2]),
        idx_flood=np.array([2, 1]),
    )

    assert filename.is_file()
