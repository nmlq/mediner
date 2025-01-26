import os
from mediner import commands


def test_convert_csv_to_label_studio_no_file_output(mock_input_csv):
    assert commands.convert_csv_to_label_studio(mock_input_csv, "ReportText")


def test_convert_csv_to_label_studio_with_file_output(
        mock_input_csv,
        tmp_path):
    """Test converting the csv to the label studio format
    """
    temp_output_path = tmp_path / "temp.output.json"
    assert not os.path.isfile(str(temp_output_path))
    assert commands.convert_csv_to_label_studio(
        mock_input_csv,
        "ReportText",
        output_filename=str(temp_output_path)
    )
    assert os.path.isfile(str(temp_output_path))


def test_train():
    """Test the train function
    """