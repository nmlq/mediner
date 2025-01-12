import hashlib


def text_to_md5(text: str) -> str:
    """Convert a text input into a md5 string

    :returns str: md5 hex string
    """
    md5 = hashlib.md5()
    md5.update(text.encode())
    return md5.hexdigest()
