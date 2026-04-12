# PyPI publish — do these once, then push tag `v0.1.0`

Do **steps 1–2 before** pushing the tag (or re-run the failed workflow after fixing).

## 1. GitHub environment `pypi`

1. Open: [Environments](https://github.com/Saibabu7770/bitcal-tts/settings/environments)
2. **New environment** → name: **`pypi`** (exact spelling)
3. Save (optional: add protection rules)

## 2. PyPI trusted publisher

1. Log in: [pypi.org](https://pypi.org/account/login/)
2. Open: [Publishing settings](https://pypi.org/manage/account/publishing/) (or your project → **Publishing** after the project exists)
3. **Add a new pending publisher**:
   - **PyPI project name:** `bitcal-tts`
   - **Owner:** `Saibabu7770`
   - **Repository:** `bitcal-tts`
   - **Workflow filename:** `publish-pypi.yml`
   - **Environment name:** `pypi`

## 3. Release

After the above, either push tag `v0.1.0` or in GitHub: **Actions** → **Publish to PyPI** → **Run workflow**.

Install check: `pip install bitcal-tts`
