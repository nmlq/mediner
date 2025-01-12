import pytest
import pandas


@pytest.fixture(scope="session")
def mock_input_data() -> list[dict]:
    return [
        {
            "Sta3n": 123,
            "SiteAccessionNumber": "NULL",
            "ReportText": "Some text"
        },
        {
            "Sta3n": 456,
            "SiteAccessionNumber": "111-222222-3333",
            "ReportText": "Some more text"
        }
    ]


@pytest.fixture(scope="session")
def mock_input_df(mock_input_data) -> pandas.DataFrame:
    return pandas.DataFrame(mock_input_data)


@pytest.fixture(scope="session")
def mock_input_csv(mock_input_df, tmp_path_factory):
    mock_input_csv_path = tmp_path_factory.mktemp(
        "mock-data-mediner"
    ) / "mock-input.csv"
    mock_input_df.to_csv(mock_input_csv_path, index=False)
    return str(mock_input_csv_path)
