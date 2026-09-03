# Start here: reviewer and supervisor verification

This repository supports **two different activities**:

1. **Audit the released manuscript evidence without rerunning the experiments.** This is the recommended first step for reviewers, editors, and supervisors.
2. **Optionally reproduce the computational workflow from public data.** This is a deeper and substantially longer exercise described in `README.md`.

A full rerun is **not required** to check that the released files are intact and that the machine-readable evidence agrees with the manuscript-facing reference values. The repository includes cryptographic hashes, fixed fold assignments, frozen-panel records, machine-readable Supplementary Data, compact verification outputs, and deterministic checking scripts.

## Fastest reviewer path

From a terminal opened in the repository root:

```bash
python -m pip install -r requirements-verify.txt
python scripts/verify_repository.py --with-tests
```

Expected final status for the v1.2.5 release:

```text
PASS  Release metadata consistency
PASS  Cryptographic file-manifest integrity
PASS  Machine-readable paper/archive verification
PASS  Deterministic unit/smoke tests

REPOSITORY VERIFICATION PASS
```

The component checks currently report:

- exact non-transient release-manifest coverage;
- **22/22** manuscript/archive checks passing; and
- **8/8** deterministic tests passing.

These checks do **not** download GEO/STRING/TCGA data and do **not** rerun the feature-selection, classifier, repeated-CV, LOCO, or external-validation experiments.

## Windows: one-click verification

Double-click:

`VERIFY_REPOSITORY_WINDOWS.bat`

The script creates an isolated local environment named `.venv_review`, installs the verification dependencies there, and runs the same repository checks. It does not alter manuscript results or source outputs.

## macOS / Linux

From the repository root:

```bash
bash verify_repository_unix.sh
```

The script creates `.venv_review`, installs the verification dependencies, and runs the verification suite.

## What each check means

### 1. Release metadata consistency

```bash
python scripts/verify_release_metadata.py
```

Checks that the release number/date are synchronized across the README, `CITATION.cff`, `config.yaml`, package metadata, release notes, package marker, and release summary.

### 2. File-integrity verification

```bash
python scripts/verify_file_manifest.py
```

Recomputes SHA-256 digests and file sizes and compares them with `FILE_MANIFEST_SHA256.csv`. This detects missing, altered, duplicated, unsafe, or unexpected release files. Runtime caches are deliberately excluded.

### 3. Manuscript-evidence verification

```bash
python scripts/verify_paper_archive.py
```

Checks machine-readable evidence supporting the paper, including the primary fold assignment, frozen top-20 panel identity, Nogueira stability values, repeated matched stability inference, LASSO nonzero auditing, fold-fitted stability values, degree-preserving and topology-permutation controls, RWR-DEG reference values, external datasets, and TCGA-BRCA pair/panel checks.

### 4. Unit/smoke tests

```bash
python -m pytest -q
```

Exercises deterministic functions and helpers, including Borda ranking, pairwise Jaccard summaries, the Nogueira stability estimator, classification metrics, and TCGA helper behavior.

## What these checks do not prove

Passing integrity and unit checks is not a substitute for an independent scientific replication. It demonstrates that the released evidence is intact, internally verifiable, and consistent with the checked manuscript-facing reference values, and that the tested code paths behave as expected.

A reviewer who wants independent end-to-end reproduction can follow the **Core analysis** and **Additional analyses reported in the manuscript** sections of `README.md` to download/reconstruct public resources and rerun the workflow.

## GitHub use

The verification commands can be run after either:

- `git clone` of the repository; or
- downloading the GitHub release/ZIP and extracting it locally.

A Bash shell is **not required on Windows**. PowerShell, Command Prompt, Anaconda Prompt, Git Bash, or the included `.bat` launcher can all be used as long as Python is installed.
