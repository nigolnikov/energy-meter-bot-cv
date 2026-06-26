import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


def resolve_env_variables(raw_text: str) -> str:
    def replace_env_var(match: re.Match) -> str:
        var_name = match.group(1)
        value = os.getenv(var_name)

        if value is None:
            raise ValueError(f"Environment variable '{var_name}' is not set")

        return value

    return _ENV_PATTERN.sub(replace_env_var, raw_text)


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    load_dotenv()

    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        raw_config = file.read()

    resolved_config = resolve_env_variables(raw_config)

    return yaml.safe_load(resolved_config)
