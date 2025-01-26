from mediner import types


def test_task_to_from_dictionary():
    """Test creating a task to and from a dictionary

    :return None:
    :raises AssertionError:
    """
    task = types.Task(
        data=types.Data(
            text="Some input textls"
        ),
        meta=types.Meta(
            md5="deadbeef"
        ),
        predictions=[
            types.Prediction(
                model_version="1",
                score=1.0,
                result=[
                    types.EntityResult(
                        id="ab1",
                        value=[
                            types.SpanValue(
                                start=5,
                                end=7,
                                score=1.0,
                                text="inp",
                                labels=["LABEL_NAME"]
                            )
                        ]
                    )
                ]
            )
        ]
    )
    dictionary = types.asdict(task)
    expected_dictionary = {
        "data": {"text": "Some input textls"},
        "meta": {"md5": "deadbeef"},
        "predictions": [{
            "model_version": "1",
            "score": 1.0,
            "result": [{
                "id": "ab1",
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": [{
                    "start": 5,
                    "end": 7,
                    "score": 1.0,
                    "text": "inp",
                    "labels": ["LABEL_NAME"]
                }]
            }]
        }]
    }
    assert dictionary and dictionary == expected_dictionary
    assert types.Task.from_dict(dictionary) == task
