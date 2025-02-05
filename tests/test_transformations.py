from mediner import transformations
from mediner import types
from spacy.tokens import DocBin


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


def test_files_to_tasks(mock_label_studio_export_json_filename):
    """Test making tasks from json input format

    :return None:
    :raises AssertionError:
    """
    tasks = transformations.files_to_tasks([mock_label_studio_export_json_filename])
    assert tasks and all([isinstance(task, types.Task) for task in tasks])


def test_tasks_to_docbin(mock_label_studio_export_json_filename):
    """Test making a md5 hash string from an empty input

    :return None:
    :raises AssertionError:
    """
    tasks = transformations.files_to_tasks([mock_label_studio_export_json_filename])
    docbin = transformations.tasks_to_docbin(tasks)
    assert docbin and isinstance(docbin, DocBin) and len(docbin) == len(tasks)
