# type: ignore
import nox


@nox.session(venv_backend="conda")
def lint(session):
    session.install("--upgrade", "setuptools", "pip")
    session.install("-r", "requirements/dev.txt")
    session.run("flake8", "graphlog")
    session.run("black", "--check", "graphlog")
    session.run("mypy", "--strict", "graphlog")
