# GitHub upload checklist

Use this checklist when replacing the current GitHub repository contents.

1. Extract `NIBFS_GITHUB_READY_TO_UPLOAD.zip` on your computer.
2. Open the existing local repository in GitHub Desktop and choose **Repository → Show in Explorer**.
3. Keep the hidden `.git` folder. Do **not** delete `.git`.
4. Remove the old/wrong repository files, then copy the **contents** of the extracted `NIBFS-reproducibility` folder into the repository root.
5. The repository root should directly contain `README.md`, `config.yaml`, `src/`, `notebooks/`, `supplementary_data/`, `results/`, `scripts/`, and `tests/`. There should not be an extra `NIBFS-reproducibility/NIBFS-reproducibility/` nesting level.
6. Do not upload the outer ZIP file to the repository.
7. In GitHub Desktop, review the Changes tab. Confirm that `src/` and `notebooks/` are visible.
8. Commit with a message such as `Paper-facing NIBFS reproducibility release v1.0.0`.
9. Click **Push origin**.
10. Open the GitHub website and verify that `README.md`, `src/`, `notebooks/`, and `supplementary_data/` are visible at the top level.

Before pushing, the package was checked with:

```bash
python scripts/verify_paper_archive.py
python scripts/verify_file_manifest.py
pytest -q
```
