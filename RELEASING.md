# Releasing `bitcal-tts` on PyPI

The package name on PyPI is **`bitcal-tts`** (import name `bitcal_tts`).

## Before the first release

1. Confirm `version` in `pyproject.toml` and `src/bitcal_tts/__init__.py` match.
2. Ensure the project name is available on [PyPI](https://pypi.org/project/bitcal-tts/). If it is taken, change `name` in `pyproject.toml` and update this doc and the publish workflow URL.

## Option A — GitHub Actions (recommended)

Uses [trusted publishing](https://docs.pypi.org/trusted-publishers/) (OpenID Connect). No long-lived PyPI token stored in GitHub.

1. On **PyPI**: your account → **Publishing** → **Add a new pending publisher** (or open the project → **Publishing**).

   - **PyPI project name:** `bitcal-tts`
   - **Owner:** your GitHub user or organization
   - **Repository name:** the repo that contains this code
   - **Workflow filename:** `publish-pypi.yml` (the file under `.github/workflows/`, not the YAML `name:` string)
   - **Environment name:** `pypi`

2. On **GitHub**: **Settings → Environments → New environment** → name it **`pypi`**.

3. Merge the workflow file `.github/workflows/publish-pypi.yml` into the default branch.

4. Tag and push:

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

   Or run the workflow manually from the **Actions** tab (**workflow_dispatch**).

## Option B — Manual upload with Twine

```bash
python -m pip install --upgrade build twine
rm -rf dist/
python -m build
twine check dist/*
twine upload dist/*
```

Use a [PyPI API token](https://pypi.org/manage/account/token/) when prompted (username `__token__`, password the token value).

## TestPyPI (optional)

```bash
twine upload --repository testpypi dist/*
```

Configure `~/.pypirc` or pass `--repository-url https://test.pypi.org/legacy/`.

## After release

- Install smoke test: `pip install bitcal-tts` and `bitcal-tts --help`
- For experiments: `pip install "bitcal-tts[research]"`
