from mediner import transformations


def test_text_to_md5():
    output = transformations.text_to_md5("Hello")
    assert output and isinstance(output, str)


def test_text_to_md5_empty():
    output = transformations.text_to_md5("")
    assert output and isinstance(output, str)
