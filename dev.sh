function zeldaRestart() {
  echo "=================>"
  killall zelda-yolo
  python main.py
}

export -f zeldaRestart

find . -name "*.py" -not -path "./venv/*" | entr -r bash -c "zeldaRestart"