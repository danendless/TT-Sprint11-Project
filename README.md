# TT-Sprint11-Project 

## Setup for Development (recommended)

### 1) Create and activate a project environment

    mamba create -n <ENV_NAME> python=3.12 pip -y
    mamba activate <ENV_NAME>

### 2) (Optional) Install heavy scientific packages via conda first

Use this only if you prefer conda-forge builds for compiled packages, or if pip install fails.

    mamba install -c conda-forge numpy pandas scipy scikit-learn pyarrow -y

### 3.1) Install this project + dependencies from pyproject.toml (editable mode install)

    python3 -m pip install -e ".[dev]"

This installs:
- Runtime dependencies
- Development tools (pytest, ruff, etc.)
- Your project in editable mode

### 3.2) Alternatively, install dev tools via pip manually and runtime dependencies from requirements.txt and anything else still missing from pyproject.toml

    python3 -m pip install -r requirements.txt
    python3 -m pip install devtool1 devtool2 missingpackage1 missingpackage2

### 4) Quick checks

    python -V
    pytest
    ruff check .

### 5) VS Code

- Open the folder in VS Code (WSL)
- Select the interpreter that matches <ENV_NAME>
- If using notebooks, select the kernel that matches <ENV_NAME> (install ipykernel in the env if needed)

Note:
`python -m ipykernel install ...` is generally only needed for standalone Jupyter (browser Jupyter Notebook/Lab), not VS Code.

---

## Deployment (Render / simple pip deploy)

Render typically installs from requirements.txt if not configured otherwise(runtime dependencies only):

    pip install -r requirements.txt

Keep requirements.txt minimal (top-level runtime dependencies only).
Development tools (pytest, ruff, ipykernel) should NOT be included.

---

## Notes

- Ruff configuration lives in pyproject.toml.
- Runtime + development dependencies are defined in pyproject.toml.
- `.[dev]` installs development extras.
- requirements.txt is for deployment/runtime installs (e.g., Render).
- VS Code workspace settings/extensions live in .vscode/
