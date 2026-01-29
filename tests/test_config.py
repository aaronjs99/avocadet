import pytest
import os
import tempfile
import yaml
from avocadet_lib.config_loader import ConfigLoader


@pytest.fixture
def config_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create a dummy config file
        data = {"key": "value", "nested": {"a": 1}}
        with open(os.path.join(tmpdirname, "test.yaml"), "w") as f:
            yaml.dump(data, f)
        yield tmpdirname


def test_config_loader(config_dir):
    loader = ConfigLoader(config_dir)

    # Loader loads specific known files like detector.yaml.
    # Since our fixture writes test.yaml, it won't be picked up by the hardcoded list.
    # We should update ConfigLoader to load generic or specific files for testing,
    # or write 'detector.yaml' in the fixture.

    # Let's verify defaults for missing files
    assert loader.get("detector") == {}


def test_config_loader_with_valid_file(config_dir):
    data = {"model_path": "foo/bar.pt"}
    with open(os.path.join(config_dir, "detector.yaml"), "w") as f:
        yaml.dump(data, f)

    loader = ConfigLoader(config_dir)
    assert loader.get("detector", "model_path") == "foo/bar.pt"


def test_update():
    loader = ConfigLoader(".")
    loader.update("section", "key", 123)
    assert loader.get("section", "key") == 123
