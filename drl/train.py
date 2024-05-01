from stable_baselines3 import PPO

from drl.custom_env import CustomGameEnv


def main():
    env = CustomGameEnv()
    model = PPO("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("ppo_cartpole")


if __name__ == "__main__":
    main()
