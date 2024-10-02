from pathlib import Path
from typing import List, Union

import yaml


def save_command(path: Union[str, Path], sysargv: List):
    if isinstance(path, str):
        path = Path(path)
    if path.suffix == "":
        path.mkdir(exist_ok=True, parents=True)
        path = path / "cmd.txt"
    elif path.suffix == ".txt":
        dirpath = path.parent
        dirpath.mkdir(exist_ok=True, parents=True)
    else:
        raise ValueError

    with open(path, "w") as f:
        cmd = ""
        for arv in sysargv:
            cmd += f"{arv} "
        f.write(f"{cmd}")

    print(f"[*] Saved command at {path}")


def save_config(path: Union[str, Path], dic):
    if isinstance(path, str):
        path = Path(path)
    if path.suffix == "":
        path.mkdir(exist_ok=True, parents=True)
        path = path / "config.yaml"
    elif path.suffix == ".yaml" or path.suffix == ".yml":
        dirpath = path.parent
        dirpath.mkdir(exist_ok=True, parents=True)
    else:
        raise ValueError

    with open(path, "w") as f:
        yaml.dump(dic, f)
