from mediner import types


def test_task_to_from_dictionary(mock_label_studio_export_json_data):
    """Test creating a task to and from a dictionary

    :return None:
    :raises AssertionError:
    """
    tasks = [types.Task(**task) for task in mock_label_studio_export_json_data]
    dicts = [task.dict() for task in tasks]
    assert tasks and dicts
    assert dicts == mock_label_studio_export_json_data
