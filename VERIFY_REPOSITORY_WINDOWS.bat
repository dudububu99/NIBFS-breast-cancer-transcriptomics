@echo off
setlocal
cd /d "%~dp0"

echo ================================================================
echo NIBFS v1.2.5 - reviewer verification (no manuscript rerun)
echo ================================================================

where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python was not found on PATH.
  echo Install Python 3.10+ or open this folder from Anaconda Prompt.
  pause
  exit /b 1
)

if not exist ".venv_review\Scripts\python.exe" (
  echo Creating isolated verification environment .venv_review ...
  python -m venv .venv_review
  if errorlevel 1 goto :fail
)

call ".venv_review\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 goto :fail
python -m pip install -r requirements-verify.txt
if errorlevel 1 goto :fail
python scripts\verify_repository.py --with-tests
if errorlevel 1 goto :fail

echo.
echo ================================================================
echo REPOSITORY VERIFICATION COMPLETED SUCCESSFULLY
echo No manuscript experiment was rerun.
echo ================================================================
pause
exit /b 0

:fail
echo.
echo ================================================================
echo REPOSITORY VERIFICATION FAILED

echo Inspect the error above before using this release.
echo ================================================================
pause
exit /b 1
