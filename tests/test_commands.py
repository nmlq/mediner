import os
from mediner import commands


def test_convert_csv_to_label_studio_no_file_output(mock_input_csv):
    """Test converting the csv to the label studio format no file output

    :return None:
    :raises AssertionError:
    """
    assert commands.convert_csv_to_label_studio(mock_input_csv, "ReportText")


def test_convert_csv_to_label_studio_with_file_output(
        mock_input_csv,
        tmp_path):
    """Test converting the csv to the label studio format

    :return None:
    :raises AssertionError:
    """
    temp_output_path = tmp_path / "temp.output.json"
    assert not os.path.isfile(str(temp_output_path))
    assert commands.convert_csv_to_label_studio(
        mock_input_csv,
        "ReportText",
        output_filename=str(temp_output_path)
    )
    assert os.path.isfile(str(temp_output_path))


def mock_model_load(*a, **k):
    class MockEnt:
        start_char = 0
        end_char = 5
        text = "mock entity"
        label_ = "mock label"

    class MockDoc:
        ents = [MockEnt()]*10

    def mock_nlp(*a, **k):
        return MockDoc()

    return mock_nlp


def test_convert_jsons_to_label_studio_with_predictions(
        mock_label_studio_export_json_filename,
        tmp_path,
        monkeypatch):
    """Test converting the jsons to the label studio format
    mock predict on json inputs

    :return None:
    :raises AssertionError:
    """
    monkeypatch.setattr(commands, 'load', mock_model_load)
    temp_output_path = tmp_path / "temp.output.json"
    assert not os.path.isfile(str(temp_output_path))
    assert commands.convert_jsons_to_label_studio(
        [mock_label_studio_export_json_filename]*3,
        output_filename=str(temp_output_path),
        predict=True,
        model_filename="mock_spacy_filename.pkl"
    )
    assert os.path.isfile(str(temp_output_path))


def test_train(
        tmp_path,
        mock_label_studio_export_json_filename,
        test_config_filename):
    """Test the train function
    """
    temp_output_model_path = tmp_path / "temp.output.model.pkl"
    temp_output_path = str(tmp_path)
    temp_output_filename = str(temp_output_model_path)
    assert not os.path.isfile(temp_output_filename)
    trained_model_output_filename = commands.train(
        input_filenames=[mock_label_studio_export_json_filename],
        output_filename=temp_output_filename,
        output_path=temp_output_path,
        config_filename=test_config_filename
    )
    assert os.path.isfile(trained_model_output_filename)
    assert trained_model_output_filename == temp_output_filename


def test_train_k_2(
        tmp_path,
        mock_label_studio_export_json_filename,
        test_config_filename):
    """Test the train function with k=2
    """
    temp_output_model_path = tmp_path / "temp.output.model.pkl"
    temp_output_path = str(tmp_path)
    temp_output_filename = str(temp_output_model_path)
    assert not os.path.isfile(temp_output_filename)
    trained_model_output_filename = commands.train(
        input_filenames=[mock_label_studio_export_json_filename],
        output_filename=temp_output_filename,
        output_path=temp_output_path,
        config_filename=test_config_filename,
        k=2
    )
    assert os.path.isfile(trained_model_output_filename)
    assert trained_model_output_filename == temp_output_filename

def test_train_hold_out_percentage_30(
        tmp_path,
        mock_label_studio_export_json_filename,
        test_config_filename):
    """Test the train function with percentage=0.3
    """
    temp_output_model_path = tmp_path / "temp.output.model.pkl"
    temp_output_path = str(tmp_path)
    temp_output_filename = str(temp_output_model_path)
    assert not os.path.isfile(temp_output_filename)
    trained_model_output_filename = commands.train(
        input_filenames=[mock_label_studio_export_json_filename],
        output_filename=temp_output_filename,
        output_path=temp_output_path,
        config_filename=test_config_filename,
        percentage=0.3
    )
    assert os.path.isfile(trained_model_output_filename)
    assert trained_model_output_filename == temp_output_filename