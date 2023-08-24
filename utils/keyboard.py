import time
import pynput
from colorama import Fore, Style

from config.const import ONCE_DURATION

ctr = pynput.keyboard.Controller()


def press_key(key, duration=0.1):
    if duration < 0.1:
        duration = 0.1
    if duration > ONCE_DURATION:
        duration = ONCE_DURATION

    print(Fore.GREEN + "Pressing {} for {} seconds.".format(key, str(duration)) + Style.RESET_ALL)

    for k in key:
        if k == '':
            continue
        ctr.press(k)

    time.sleep(duration)

    for k in key:
        if k == '':
            continue
        ctr.release(k)

    time.sleep(0.1)
