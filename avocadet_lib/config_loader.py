import os
import yaml
from typing import Dict, Any, Optional


class ConfigLoader:
    """
    Loads and merges configuration from YAML files.
    """

    def __init__(self, config_dir: str):
        self.config_dir = config_dir
        self.config = {}
        self._load_all()

    def _load_all(self):
        """Loads all known config files."""
        files = [
            "camera.yaml",
            "geometry.yaml",
            "detector.yaml",
            "runtime.yaml",
            "ros_topics.yaml",
            "visualization.yaml",
            "tiling.yaml",
            "backends.yaml",
        ]

        for f in files:
            path = os.path.join(self.config_dir, f)
            key = os.path.splitext(f)[0]
            if os.path.exists(path):
                try:
                    with open(path, "r") as stream:
                        self.config[key] = yaml.safe_load(stream) or {}
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    self.config[key] = {}
            else:
                print(f"Warning: Config file {path} not found.")
                self.config[key] = {}

    def get(self, section: str, key: str = None, default: Any = None) -> Any:
        """
        Retrieve a config value.
        Usage: get("detector", "confidence_threshold", 0.5)
               get("detector") -> returns whole dict
        """
        sec = self.config.get(section, {})
        if key is None:
            return sec
        return sec.get(key, default)

    def update(self, section: str, key: str, value: Any):
        """Updates a configuration value in memory."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value

    def print_config(self):
        """Prints the current configuration."""
        print("--- Avocadet Configuration ---")
        print(yaml.dump(self.config, default_flow_style=False))
        print("------------------------------")
