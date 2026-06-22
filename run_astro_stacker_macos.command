#!/bin/bash
cd "$(dirname "$0")"
exec "$PWD/.venv-astrostacker/bin/python3" "$PWD/astro_stacker_app.py"
