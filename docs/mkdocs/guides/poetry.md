# Installation using Poetry

You can use a Poetry-based installation if you are using
D-FAST Transverse Flow from a local clone of the Github repository,
for example if you intend to contribute to the code.

## Clone the GitHub repo
Use your own preferred way of cloning the GitHub repository of D-FAST Transverse Flow.
In the examples below it is placed in `C:\checkouts\D-FAST_transverse_flow`.

## Use Poetry to install D-FAST Transverse Flow
We use `poetry` to manage our package and its dependencies.

!!! note
    If you use `conda`, do not combine conda virtual environments with the poetry virtual environment.
    In other words, run the `poetry install` command from the `base` conda environment.

1. Download + installation instructions for Poetry are [here](https://python-poetry.org/).
2. Verify that you have Python installed, and that it matches the required version for D-FAST Transverse Flow.

  D-FAST Transverse Flow requires **Python 3.11**.  
  Before proceeding, check your Python version by running:

  ```shell
  python --version
  ```

  If your Python version is not 3.11.x, please install Python 3.11 before continuing.  
  You can find installation instructions at [python.org](https://www.python.org/downloads/).

3. After installation of Poetry itself, now use it to install your local clone of the D-FAST Transverse Flow package, as follows.
   Make sure Poetry is available on your `PATH` and run `poetry install` in the D-FAST Transverse Flow directory in your shell of choice.
   This will create a virtual environment in which D-FAST Transverse Flow is installed and made available for use in your own scripts.
   For example in an Anaconda PowerShell:
```
(base) PS C:\checkouts\D-FAST_transverse_flow> poetry install
Creating virtualenv d-fast-transverse-flow-kHkQBdtS-py3.11 in C:\Users\<username>\AppData\Local\pypoetry\Cache\virtualenvs
Installing dependencies from lock file

Package operations: 67 installs, 0 updates, 0 removals

  * Installing six (1.16.0)
[..]
Installing the current project: d-fast-transverse-flow (X.Y.Z)
(base) PS C:\checkouts\D-FAST_transverse_flow>
```  
   If you need to use an already existing Python installation, you can activate it and run `poetry env use system` before `poetry install`.

4. Test your installation, by running the D-FAST Transverse Flow pytest suite via poetry:

This step verifies that your environment and dependencies are set up correctly.  
A **successful test run** will look like this:

```
(base) PS C:\checkouts\D-FAST_transverse_flow> poetry run pytest --disable-warnings
====================== test session starts ======================
platform win32 -- Python 3.11.12, pytest-7.4.4, pluggy-1.6.0
cachedir: .pytest_cache
rootdir: C:\gdrive\algorithms\deltares\D-FAST_transverse_flow
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.12.0, pyfakefs-5.10.2, cov-6.3.0, teamcity-messages-1.33
collected 1 item                                                                                               

tests\test_cli.py ............                             [100%]

========= 1 passed, 0 deselected, 1 warnings in 2.59s =========
(base) PS C:\checkouts\D-FAST_transverse_flow>
```

If all tests pass, your D-FAST Transverse Flow installation is working correctly.

If there are **failing tests**, pytest will display detailed information about the failures.  
For example:

```
(base) (d-fast-transverse-flow-apspvf-py3.11) PS C:\checkouts\D-FAST_transverse_flow> pytest tests -m "not e2e" --disable-warnings
====================== test session starts ======================
platform win32 -- Python 3.11.9, pytest-7.4.4, pluggy-1.6.0
rootdir: C:\checkouts\D-FAST_transverse_flow
configfile: pyproject.toml
plugins: anyio-4.6.2.post1, cov-6.2.1, typeguard-4.0.0
collected 143 items / 6 deselected / 137 selected                                                                                           

tests\test_cli.py ............                             [100%]

============================ FAILURES ===========================
____________________________ test_cli ___________________________

    def test_cli():
        """Test the CLI command with the NVO Maas example."""
        # config = "examples/c04-nvo-maas/config.ini"
    
        cmd = [
            sys.executable,
            "-m",
            "dfasttf",
>           "--config", str(config),
        ]
E       NameError: name 'config' is not defined

tests\test_cli.py:20: NameError
=================================================

FAILED tests/test_cli.py::test_cli - NameError: name 'config' is not defined
================================ 1 failed in 0.32s ================================
```

If you see failures, review the error messages and traceback to help diagnose and fix issues in your installation or code.  
You may need to check your Python version, dependencies, or environment configuration.

**Tip:**  
If you want to run only a subset of tests or exclude certain tests (e.g., end-to-end tests), you can use pytest markers:
```
poetry run pytest -m "not e2e"
```
This will run all tests except those marked as `e2e`.

5. Start using D-FAST Transverse Flow. You can launch your favourite editor (for example VS Code)
by first starting a poetry shell with the virtual D-FAST Transverse Flow environment:
```
(base) PS C:\checkouts\D-FAST_transverse_flow> poetry shell
(base) PS C:\checkouts\D-FAST_transverse_flow> code
```