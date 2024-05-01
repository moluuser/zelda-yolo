import gym
import numpy as np
from ultralytics import YOLO

from config.const import WINDOWS_OWNER
from utils.window import show_window

YOLO_BEST_PATH = "../yolo/detect/train/weights/best.pt"
model = YOLO(YOLO_BEST_PATH)


class CustomGameEnv(gym.Env):
    def __init__(self):
        # Init game environment
        print("init")
        show_window(WINDOWS_OWNER, model)
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(500, 500, 3), dtype=np.uint8)

    def reset(self):
        # 重置游戏状态并返回初始观察
        print("reset")
        return np.zeros((500, 500, 3), dtype=np.uint8)

    def step(self, action):
        # 执行动作，返回下一个观察、奖励、是否终止和额外信息
        print("step")
        return np.zeros((500, 500, 3), dtype=np.uint8), 0, False, {}

    def render(self):
        # Render the environment to the screen
        print("render")
