from MarioKartEnv import EnvReceiver, EnvWrapper, DiscreteAction
import keyboard
from threading import Thread
import time

class KeyboardController:
    def __init__(self):
        self.current_action = 0  # NO_OP by default
        self._running = True
        self.thread = Thread(target=self._keyboard_monitor, daemon=True)
        self.thread.start()

    def _keyboard_monitor(self):
        while self._running:
            # Base actions
            if keyboard.is_pressed('space'):
                self.current_action = 2  # BRAKE
            elif keyboard.is_pressed('s'):
                self.current_action = 3  # BACK_UP
            # Combined up + direction controls
            elif keyboard.is_pressed('up') and keyboard.is_pressed('left'):
                self.current_action = 5  # LEFT
            elif keyboard.is_pressed('up') and keyboard.is_pressed('right'):
                self.current_action = 9  # RIGHT
            # Single direction controls
            elif keyboard.is_pressed('left'):
                self.current_action = 7  # EXTREME_LEFT
            elif keyboard.is_pressed('right'):
                self.current_action = 11  # EXTREME_RIGHT
            elif keyboard.is_pressed('up'):
                self.current_action = 1  # STRAIGHT
            else:
                self.current_action = 0  # NO_OP
            time.sleep(0.05)

    def get_action(self):
        return self.current_action

    def stop(self):
        self._running = False
        self.thread.join()

env = EnvReceiver()
env = EnvWrapper(env)
env = DiscreteAction(env)
observation = env.reset()
controller = KeyboardController()

running = True

while running:

    action = controller.get_action()

    observation, reward, done, _, info = env.step(action)

env.close()
