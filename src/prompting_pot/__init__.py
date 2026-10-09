from doctest import debug
from logging import DEBUG


def main() -> None:
    import uvicorn

    uvicorn.run("prompting_pot.app:app", host="0.0.0.0", port=8000, reload=True, log_level=DEBUG)
