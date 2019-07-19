import pyautogui
import time

# The mouse cursor to the upper-left corner of the screen will cause PyAutoGUI to raise the pyautogui.FailSafeException
# The fail-safe feature will stop the program if you quickly move the mouse far up and left
pyautogui.FAILSAFE = True

# The sleep command gives us time to bring our desired window to a default stage, after
# which the autopygui function takes over.
time.sleep(10)

# Setting a variable delay to allow chromaTOF to perform tasks.
DELAY = 0.5


# Defining a function for using navigation keys, with parameters type for the direction
# and num for number of times the key needs to be pressed.
def press_nav_arrow(type, num):
    for i in range(0, num):
        pyautogui.typewrite([type])
        time.sleep(DELAY)


# Defining a function to export the peak table, with a parameter compound number,
# to allow the program to exit when completed

def export_peaktable(compd_num):
    pyautogui.moveTo(451, 179, duration=DELAY)
    time.sleep(DELAY)
    pyautogui.click(button='left')
    time.sleep(DELAY)
    pyautogui.moveTo(1059, 513, duration=DELAY)
    pyautogui.click(button='right')
    time.sleep(DELAY)
    press_nav_arrow('down', 5)
    time.sleep(DELAY)
    pyautogui.press('enter')
    time.sleep(DELAY)
    pyautogui.press('enter')
    time.sleep(DELAY)
    pyautogui.press('enter')
    time.sleep(DELAY)
    pyautogui.press('enter')

    counter = 0
    counter_name = 1
    for i in range(0, compd_num):
        pyautogui.moveTo(451, 179, duration=DELAY)
        time.sleep(DELAY)
        pyautogui.click(button='left')
        time.sleep(DELAY)
        if counter != 38:
            counter = counter + 1
        counter_name = counter_name + 1
        press_nav_arrow('down', counter)
        time.sleep(DELAY)
        pyautogui.moveTo(1059, 513, duration=DELAY)
        pyautogui.click(button='right')
        time.sleep(DELAY)
        press_nav_arrow('down', 5)
        time.sleep(DELAY)
        pyautogui.press('enter')
        time.sleep(DELAY)
        pyautogui.press('enter')
        time.sleep(DELAY)
        pyautogui.press('backspace')
        time.sleep(DELAY)
        pyautogui.typewrite(str(counter_name))
        time.sleep(DELAY)
        pyautogui.press('enter')


export_peaktable(10000)
