@echo off

REM Set the name of the virtual environment
set VENV_NAME=myvenv

REM Create the virtual environment
python -m venv %VENV_NAME%

REM Activate the virtual environment
call %VENV_NAME%\Scripts\activate

REM Install packages from requirements.txt
pip install -r requirements.txt

python .\src\main.py

REM Deactivate the virtual environment
deactivate