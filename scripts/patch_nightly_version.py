from datetime import date

import toml


def get_nightly_version():
    today = date.today()
    return f"{today.year}.{today.month}.{today.day}"


CARGO_FILE = "Cargo.toml"
PYPROJECT_FILE = "pyproject.toml"


cargo = toml.load(CARGO_FILE)
cargo["package"]["version"] = get_nightly_version()

print(cargo)

with open(CARGO_FILE, "w") as f:
    toml.dump(cargo, f)

pyproject = toml.load(PYPROJECT_FILE)
pyproject["project"]["name"] = "torchft-nightly"

print(pyproject)

with open(PYPROJECT_FILE, "w") as f:
    toml.dump(pyproject, f)
