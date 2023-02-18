"""test __init__
    """
from personalization import __version__


def test_version():
    """test __version__ value"""
    assert __version__ == "0.0.1"
