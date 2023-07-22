function zeldaRestart() {
  echo "=================>"
  killall zelda-yolo
  python main.py
}

export -f zeldaRestart

find . -name "*.py" | entr -r bash -c "zeldaRestart"