from __future__ import annotations
import importlib
import os
import shutil
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)

# R/limma is installed before rpy2.
if shutil.which("R") is None or shutil.which("Rscript") is None:
    run(["apt-get", "update", "-qq"])
    run(["apt-get", "install", "-y", "-qq", "r-base-core", "r-base-dev"])

os.environ.setdefault("RPY2_CFFI_MODE", "ABI")
required = {
    "lightgbm": "lightgbm>=4.0,<5",
    "networkx": "networkx>=3.0",
    "joblib": "joblib>=1.3",
    "yaml": "pyyaml>=6",
    "neuroCombat": "neuroCombat==0.2.12",
    "gprofiler": "gprofiler-official>=1.0.0",
    "rpy2": "rpy2[numpy]==3.6.6",
}
missing = []
for module, package in required.items():
    try:
        importlib.import_module(module)
    except Exception:
        missing.append(package)
if missing:
    run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--upgrade-strategy", "only-if-needed", *missing])
    importlib.invalidate_caches()

r_code = r'''options(repos=c(CRAN='https://cloud.r-project.org'))
if (!requireNamespace('BiocManager', quietly=TRUE)) install.packages('BiocManager', quiet=TRUE)
if (!requireNamespace('limma', quietly=TRUE)) BiocManager::install('limma', ask=FALSE, update=FALSE, quiet=TRUE)
stopifnot(requireNamespace('limma', quietly=TRUE))
cat('limma version:', as.character(packageVersion('limma')), '\n')
'''
run(["R", "--vanilla", "-q", "-e", r_code])

for module in ["numpy", "pandas", "scipy", "sklearn", "matplotlib", *required.keys()]:
    importlib.import_module(module)
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
import numpy as np
with localconverter(ro.default_converter + numpy2ri.converter):
    ro.globalenv["nibfs_env_test_x"] = np.asarray([1.0, 2.0, 3.0], dtype=float)
assert float(ro.r("sum(nibfs_env_test_x)")[0]) == 6.0
ro.r("suppressPackageStartupMessages(library(limma))")
print("ENVIRONMENT CHECK: PASS (rpy2 numpy2ri + limma)")
