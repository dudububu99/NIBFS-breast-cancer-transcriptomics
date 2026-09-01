# Software environment evidence

The executed analysis notebook explicitly recorded:

- NumPy 2.0.2
- pandas 2.2.2
- R 4.6.0
- limma 3.68.4

The executed notebook did not print exact patch versions for every Python dependency. The repository therefore preserves executable version bounds instead of retroactively assigning unverified patch versions. See `requirements.txt`, `requirements-colab.txt`, and `install_environment_colab.py`. The supplementary-analysis Colab runner pins `rpy2[numpy]==3.6.6` and `neuroCombat==0.2.12` to reproduce the Python-R bridge used for the added strict analyses.
