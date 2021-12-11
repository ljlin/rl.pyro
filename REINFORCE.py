import gym
import matplotlib.pyplot as plt
import numpy
import torch
import pyro
import tqdm

import utils.common
import utils.envs
import utils.seed
import utils.torch

import warnings

warnings.filterwarnings("ignore")


class REINFORCE(torch.nn.Module):
    """
    CS885 Fall 2021 - Reinforcement Learning
    https://cs.uwaterloo.ca/~ppoupart/teaching/cs885-fall21/schedule.html

    - MODE = hard: REINFORCE Algorithm (Vanilla Policy Gradient)
      Slides 10
      https://cs.uwaterloo.ca/~ppoupart/teaching/cs885-fall21/slides/cs885-lecture7a.pdf
      https://spinningup.openai.com/en/latest/algorithms/vpg.html
    - MODE = soft: Soft REINFORCE (pure PyTroch)
      Sergey Levine. Reinforcement learning and control as probabilistic inference: Tutorial and review.
      CoRR, abs/1805.00909, 2018. URL http://arxiv.org/abs/1805.00909.
    - MODE = pyro: Soft REINFORCE (pure Pyro)
    """

    def __init__(
            self,
            MODE,
            SMOKE_TEST=False,
            ENV_NAME="CartPole-v0",
            GAMMA=0.99,
            # Discount factor in episodic reward objective
            MINIBATCH_SIZE=64,
            # How many examples to sample per train step
            HIDDEN=512,
            # Hiddien states
            LEARNING_RATE=5e-4,
            # Learning rate for Adam optimizer
            SEEDS=[1, 2, 3, 4, 5],
            # Randoms seeds for mutiple trails
            EPISODES=300 * 25,
            # Total number of episodes to learn over
            TEMPERATURE=None,
            PRIOR=None,
            MODEL_MODE=None
    ):
        super().__init__()
        self.t = utils.torch.TorchHelper()
        # Constants
        self.DEVICE = self.t.device
        self.ENV_NAME = ENV_NAME
        self.GAMMA = GAMMA
        self.MINIBATCH_SIZE = MINIBATCH_SIZE
        self.LEARNING_RATE = LEARNING_RATE
        self.HIDDEN = HIDDEN
        self.MODE = MODE

        if SMOKE_TEST:
            self.SEEDS = [1, 2]
            self.EPISODES = 20
        else:
            self.SEEDS = SEEDS
            self.EPISODES = EPISODES

        assert (self.SOFT_OFF != self.SOFT_ON)
        assert (self.SOFT_ON == (TEMPERATURE is not None))
        assert (self.SOFT_OFF == (TEMPERATURE is None))
        self.TEMPERATURE = TEMPERATURE

        assert (self.SVI_ON == (PRIOR is not None))
        assert (self.SVI_ON == (MODEL_MODE is not None))
        if self.SVI_ON:
            assert (PRIOR is not None)
            self.PRIOR = PRIOR
            self.prior = getattr(self, f"prior_{PRIOR}", None)
            assert (self.prior is not None)

            assert (MODEL_MODE is not None)
            self.MODEL_MODE = MODEL_MODE
            self.model = getattr(self, f"model_{MODEL_MODE}", None)
            assert (self.model is not None)

    @property
    def SVI_ON(self):
        return self.MODE == "pyro"

    @property
    def SOFT_ON(self):
        return self.MODE == "pyro" or self.MODE == "soft"

    @property
    def SOFT_OFF(self):
        return self.MODE == "hard"

    def create_everything(self, seed):
        utils.seed.seed(seed)
        env = gym.make(self.ENV_NAME)
        env.seed(seed)
        test_env = gym.make(self.ENV_NAME)
        test_env.seed(10 + seed)

        assert (isinstance(env.action_space, gym.spaces.discrete.Discrete))
        assert (isinstance(env.observation_space, gym.spaces.box.Box))
        self.OBS_N = env.observation_space.shape[0]
        self.ACT_N = env.action_space.n
        self.unif = torch.ones(self.ACT_N, device=self.t.device) / self.ACT_N

        self.pi = torch.nn.Sequential(
            torch.nn.Linear(self.OBS_N, self.HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(self.HIDDEN, self.HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(self.HIDDEN, self.ACT_N),
            torch.nn.Softmax(dim=-1)
        ).to(self.DEVICE)

        if self.SVI_ON:
            adma = pyro.optim.Adam({"lr": self.LEARNING_RATE})
            OPT = pyro.infer.SVI(self.model, self.guide, adma, loss=pyro.infer.Trace_ELBO())
        else:
            OPT = torch.optim.Adam(self.pi.parameters(), lr=self.LEARNING_RATE)

        return env, test_env, self.pi, OPT

    def guide(self, env=None, trajectory=None):
        pyro.module("agentmodel", self)
        step = 0
        S, A, R, D = [], [], [], []
        obs = env.reset()
        done = False
        while not done:
            S.append(obs)
            D.append(done)
            action = pyro.sample(
                f"action_{step}",
                pyro.distributions.Categorical(self.pi(self.t.f(obs)))
            ).item()
            obs, reward, done, info = env.step(action)
            A.append(action)
            R.append(reward)
            step += 1
        S.append(obs)
        D.append(done)

        trajectory["S"] = self.t.f(S)
        trajectory["A"] = self.t.i(A)
        trajectory["R"] = self.t.f(R)
        trajectory["D"] = self.t.b(D)

    def prior_pi(self, state):
        return self.pi(state)

    def prior_unif(self, state):
        return self.unif

    def model_sequential(self, env=None, trajectory=None):
        S, R = trajectory["S"], trajectory["R"]
        for step, state in enumerate(S[:-1]):
            action = pyro.sample(
                "action_{}".format(step),
                pyro.distributions.Categorical(self.prior(state))
            )
            pyro.factor(f"reward_{step}", R[step] / self.TEMPERATURE)

    def model_plate(self, env=None, trajectory=None):
        S, R = trajectory["S"], trajectory["R"]
        for step in pyro.plate("trajectory", len(R)):
            action = pyro.sample(
                f"action_{step}",
                pyro.distributions.Categorical(self.prior(S[step]))
            )
            pyro.factor(f"reward_{step}", torch.sum(R / self.TEMPERATURE))

    def policy(self, env, obs):
        with torch.no_grad():
            obs = self.t.f(obs).view(-1, self.OBS_N)  # Convert to torch tensor
            action = torch.distributions.Categorical(self.pi(obs)).sample().item()
        return action

    def train(self, seed):

        print("Seed=%d" % seed)
        env, test_env, pi, OPT = self.create_everything(seed)

        if self.SVI_ON:
            pyro.clear_param_store()

        trainRs = []
        last25Rs = []
        print("Training:")
        pbar = tqdm.trange(self.EPISODES)
        for epi in pbar:
            if self.SVI_ON:
                trajectory = {"S": None, "R": None}
                OPT.step(env, trajectory=trajectory)
                trainRs += [sum(trajectory["R"]).item()]
            else:
                # Play an episode and log episodic reward

                S, A, R = utils.envs.play_episode_tensor(env, self.policy, self.t)

                nSteps = len(S)

                G = torch.zeros(nSteps, device=self.t.device)
                G[-1] = R[-1]
                for step in reversed(range(nSteps - 1)):
                    G[step] = R[step] + self.GAMMA * G[step + 1]

                log_prob = torch.log(pi(S[:-1]).gather(-1, A.view(-1, 1))).squeeze()
                with torch.no_grad():
                    if self.SOFT_ON:
                        G[:-1] -= self.TEMPERATURE * log_prob
                    gamma_n = torch.pow(self.GAMMA, torch.arange(nSteps - 1, device=self.t.device))
                loss = - gamma_n * G[:-1] * log_prob
                OPT.zero_grad()
                loss.mean().backward()
                OPT.step()

                trainRs += [sum(R).item()]
                # Update progress bar
            last25Rs += [sum(trainRs[-25:]) / len(trainRs[-25:])]
            pbar.set_description("R25(%g)" % (last25Rs[-1]))

        # Close progress bar, environment
        pbar.close()
        print("Training finished!")
        env.close()
        test_env.close()

        return last25Rs

    def run(self, info=None, SHOW=True):
        # Train for different seeds
        label=f"REINFORCE-{self.MODE}"
        if self.SVI_ON:
            label += f"-prior_{self.PRIOR}-model_{self.MODEL_MODE}"
        if self.SOFT_ON:
            label += f"-TEMPERATURE({self.TEMPERATURE})"
        filename = utils.common.safe_filename(
            f"{label}-{self.ENV_NAME}{'-' + info + '-' if info else '-'}-SEED({self.SEEDS})")
        print(filename)
        utils.common.train_and_plot(
            self.train,
            self.SEEDS,
            filename,
            label,
            range(self.EPISODES),
            SHOW
        )


if __name__ == "__main__":
    reinforece = REINFORCE(
        "pyro",
        SMOKE_TEST=True,
        TEMPERATURE=1,
        PRIOR="unif",
        MODEL_MODE="plate"
    )
    reinforece.run()
