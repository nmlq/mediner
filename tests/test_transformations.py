from mediner import transformations


def test_text_to_md5():
    """Test making a md5 hash string from an input

    :return None:
    :raises AssertionError:
    """
    output = transformations.text_to_md5("Hello")
    assert output and isinstance(output, str)


def test_text_to_md5_empty():
    """Test making a md5 hash string from an empty input

    :return None:
    :raises AssertionError:
    """
    output = transformations.text_to_md5("")
    assert output and isinstance(output, str)


def test_files_to_annotations(mock_label_studio_export_json_filename):
    """Test making a md5 hash string from an empty input

    :return None:
    :raises AssertionError:
    """
    annotations = transformations.files_to_annotations([mock_label_studio_export_json_filename])
    assert annotations


#def test_annotations_to_docbin()
