# Software environment

The executed analysis notebook recorded the following core versions:

- NumPy 2.0.2
- pandas 2.2.2
- R 4.6.0
- limma 3.68.4

Exact patch versions were not printed for every Python dependency during the archived execution. The repository therefore preserves executable version bounds rather than assigning unverified patch versions retrospectively. See `requirements.txt`, `requirements-colab.txt`, and `install_environment_colab.py`. The supplementary-analysis environment uses `rpy2[numpy]==3.6.6` and `neuroCombat==0.2.12` for the Python-R bridge used by the fold-fitted analyses.
