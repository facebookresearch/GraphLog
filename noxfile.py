# type: ignore
import nox


def install_base_packages(session):
    session.install(
        "-f",
        "https://pytorch-geometric.com/whl/torch-1.4.0.html",
        "-f",
        "https://download.pytorch.org/whl/torch_stable.html",
        "-r",
        "requirements/base.txt",
    )


@nox.session(venv_backend="conda")
def lint(session):
    session.install("--upgrade", "setuptools", "pip")
    install_base_packages(session)
    session.install("-r", "requirements/dev.txt")
    session.run("flake8", "graphlog")
    session.run("black", "--check", "graphlog")
    session.run("mypy", "--strict", "graphlog")


@nox.session(venv_backend="conda")
def test(session) -> None:
    session.install("--upgrade", "setuptools", "pip")
    install_base_packages(session)
    session.install("-r", "requirements/dev.txt")
    session.run("pytest", "tests")
