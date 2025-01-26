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


def test_train(tmp_path, mock_label_studio_export_json, test_config_filename):
    """Test the train function
    """
    temp_output_model_path = tmp_path / "temp.output.model.pkl"
    temp_output_path = str(tmp_path)
    temp_output_filename = str(temp_output_model_path)
    assert not os.path.isfile(temp_output_filename)
    trained_model_output_filename = commands.train(
        input_filenames=[mock_label_studio_export_json],
        output_filename=temp_output_filename,
        output_path=temp_output_path,
        config_filename=test_config_filename
    )
    assert os.path.isfile(trained_model_output_filename)
    assert trained_model_output_filename == temp_output_filename
