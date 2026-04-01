import pytest
import pandas
import os
import json


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


@pytest.fixture(scope="session")
def mock_label_studio_export_json_filename():
    dirname = os.path.dirname(os.path.abspath(__file__))
    test_json_filename = f"{dirname}/test-label-studio-export.json"
    return test_json_filename


@pytest.fixture(scope="session")
def mock_label_studio_export_json_data(mock_label_studio_export_json_filename):
    with open(mock_label_studio_export_json_filename) as jf:
        data = json.load(jf)
    return data


@pytest.fixture(scope="session")
def test_config_filename():
    dirname = os.path.dirname(os.path.abspath(__file__))
    test_config_filename = f"{dirname}/tests_config.cfg"
    return test_config_filename
