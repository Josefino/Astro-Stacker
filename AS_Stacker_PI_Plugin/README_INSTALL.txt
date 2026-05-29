AS_Stacker PixInsight Script Plugin
==================================

This folder contains the PixInsight wrapper and the Python stacking engine.

Files:
- AS_Stacker_PI.js        PixInsight Feature Script wrapper
- astro_stacker_cli.py    command-line bridge used by PixInsight
- astro_stacker_app6.py   stacking/calibration/alignment engine
- requirements.txt        Python packages needed by the CLI

Install in PixInsight:
1. Open PixInsight.
2. Go to SCRIPT > Feature Scripts...
3. Click Add.
4. Select AS_Stacker_PI.js from this folder.
5. Click Done.
6. Run it from SCRIPT > Utilities > AS_Stacker.

Python:
- The wrapper calls an external Python executable.
- Use the Python from your working virtual environment.
- On this machine that has been:
  /Users/josef/Downloads/path/to/venv/bin/python3

If packages are missing, install them into that Python environment:
  /Users/josef/Downloads/path/to/venv/bin/python3 -m pip install -r requirements.txt

Notes:
- Keep all files in this folder together.
- Output files always start with AS_.
- The wrapper remembers the last used Python path, input folder, output folder, and stacking settings.
- Console output is ASCII only, because PixInsight can display accented characters incorrectly.
