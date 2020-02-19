import nox


@nox.session(venv_backend="conda")
def lint(session):
    session.install("--upgrade", "setuptools", "pip")
    session.install("-r", "requirements.txt")
    session.run("flake8", "ml_logger")
    session.run("black", "--check", "ml_logger")
    session.run("mypy", "--strict", "ml_logger")
