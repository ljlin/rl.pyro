import warnings

import gym
import pyro
import torch
import tqdm

import utils.common
import utils.envs
import utils.seed
import utils.torch

warnings.filterwarnings("ignore")


class REINFORCE:
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
            ENV_NAME,
            GAMMA,
            SMOKE_TEST=False,
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
            MODEL_MODE=None,
            USE_LOGSOFTMAX_FOR_HARD=None,
            DEVICE=None
    ):
        super().__init__()
        if DEVICE:
            self.t = utils.torch.TorchHelper(DEVICE)
            self.DEVICE = DEVICE
        else:
            self.t = utils.torch.TorchHelper()
            self.DEVICE = self.t.device

        self.ENV_NAME = ENV_NAME
        self.GAMMA = GAMMA
        self.LN_GAMMA = torch.log(self.t.f(self.GAMMA))
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

        assert ((USE_LOGSOFTMAX_FOR_HARD is not None) <= self.SOFT_OFF)
        self.USE_LOGSOFTMAX_FOR_HARD = USE_LOGSOFTMAX_FOR_HARD

    @property
    def SVI_ON(self):
        return self.MODE == "pyro"

    @property
    def SOFT_ON(self):
        return self.MODE == "pyro" or self.MODE == "soft"

    @property
    def SOFT_OFF(self):
        return self.MODE == "hard"

    @property
    def USE_LOGSOFTMAX(self):
        return self.SOFT_ON or self.USE_LOGSOFTMAX_FOR_HARD

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
        self.unif_logits = torch.ones(self.ACT_N, device=self.DEVICE).detach()

        if self.USE_LOGSOFTMAX:
            self.policy_net = torch.nn.Sequential(
                torch.nn.Linear(self.OBS_N, self.HIDDEN), torch.nn.ReLU(),
                torch.nn.Linear(self.HIDDEN, self.HIDDEN), torch.nn.ReLU(),
                torch.nn.Linear(self.HIDDEN, self.ACT_N),
                torch.nn.LogSoftmax(dim=-1)
            ).to(self.DEVICE)
        else:
            self.policy_net = torch.nn.Sequential(
                torch.nn.Linear(self.OBS_N, self.HIDDEN), torch.nn.ReLU(),
                torch.nn.Linear(self.HIDDEN, self.HIDDEN), torch.nn.ReLU(),
                torch.nn.Linear(self.HIDDEN, self.ACT_N),
                torch.nn.Softmax(dim=-1)
            ).to(self.DEVICE)

        if self.SVI_ON:
            adma = pyro.optim.Adam({"lr": self.LEARNING_RATE})
            OPT = pyro.infer.SVI(self.model, self.guide, adma, loss=pyro.infer.Trace_ELBO())
        else:
            OPT = torch.optim.Adam(self.policy_net.parameters(), lr=self.LEARNING_RATE)

        return env, test_env, self.policy_net, OPT

    def guide(self, env=None, trajectory=None):
        pyro.module("policy_network", self.policy_net)
        S, A, R, D, step = [], [], [], [], 0
        obs = env.reset()
        done = False
        while not done:
            S.append(obs)
            D.append(done)
            action = pyro.sample(
                f"action_{step}",
                pyro.distributions.Categorical(
                    logits=self.policy_net(self.t.f(obs))
                )
            ).item()
            obs, reward, done, _ = env.step(action)
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
        return self.policy_net(state)

    def prior_unif(self, state):
        return self.unif_logits

    def model_sequential(self, env=None, trajectory=None):
        S, R = trajectory["S"], trajectory["R"]
        for step, state in enumerate(S[:-1]):
            action = pyro.sample(
                "action_{}".format(step),
                pyro.distributions.Categorical(
                    logits=self.prior(state).detach()
                )
            )
            pyro.factor(f"discount_{step}", self.LN_GAMMA)
            pyro.factor(f"reward_{step}", R[step] / self.TEMPERATURE)

    def model_plate(self, env=None, trajectory=None):
        S, R = trajectory["S"], trajectory["R"]
        for step in pyro.plate("trajectory", len(R)):
            action = pyro.sample(
                f"action_{step}",
                pyro.distributions.Categorical(
                    logits=self.prior(S[step]).detach()
                )
            )
            pyro.factor(f"discount_{step}", self.LN_GAMMA)
            pyro.factor(f"reward_{step}", R[step] / self.TEMPERATURE)

    def update_network(self, S, A, R, policy_net, OPT):
        if self.USE_LOGSOFTMAX:
            log_prob = policy_net(S).gather(-1, A.view(-1, 1)).squeeze()
        else:
            log_prob = policy_net(S).gather(-1, A.view(-1, 1)).squeeze().log()

        G = torch.zeros_like(R, device=self.DEVICE)
        G[-1] = R[-1]
        for step in range(-2, - R.shape[0] - 1, -1):
            G[step] = R[step] + self.GAMMA * G[step + 1]

        with torch.no_grad():
            if self.SOFT_ON:
                G -= self.TEMPERATURE * log_prob
            gamma_n = torch.pow(self.GAMMA, torch.arange(R.shape[0], device=self.DEVICE))
        loss = - (gamma_n * G * log_prob).mean()

        OPT.zero_grad()
        loss.backward()
        OPT.step()

        return loss.item()

    def train(self, seed):

        print("Seed=%d" % seed)
        env, test_env, policy_net, OPT = self.create_everything(seed)

        if self.SVI_ON:
            pyro.clear_param_store()

        def policy(env, obs):
            with torch.no_grad():
                obs = self.t.f(obs).view(-1, self.OBS_N)  # Convert to torch tensor
                kwargs = {"logits" if self.USE_LOGSOFTMAX else "probs": policy_net(obs)}
                action = torch.distributions.Categorical(**kwargs).sample().item()
            return action

        trainRs = []
        last25Rs = []
        print("Training:")
        pbar = tqdm.trange(self.EPISODES)
        for epi in pbar:
            if self.SVI_ON:
                trajectory = {}
                OPT.step(env, trajectory=trajectory)
                trainRs += [sum(trajectory["R"]).item()]
            else:
                # Play an episode and log episodic reward
                S, A, R = utils.envs.play_episode_tensor(env, policy, self.t)
                self.update_network(S[:-1], A, R, policy_net, OPT)
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
        label = f"REINFORCE-{self.MODE}-γ({self.GAMMA})"
        if self.SVI_ON:
            label += f"-{self.PRIOR}-{self.MODEL_MODE}"
        if self.SOFT_ON:
            label += f"-λ({self.TEMPERATURE})"
        filename = utils.common.safe_filename(
            f"{label}-{self.ENV_NAME}{'-' + info + '-' if info else '-'}SEED({self.SEEDS})")
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
    pass
    # REINFORCE("hard", ENV_NAME="CartPole-v0", GAMMA=0.99, EPISODES=8000, SEEDS=[1,2,3,4,5]).run()
    # REINFORCE("hard", ENV_NAME="CartPole-v0", GAMMA=0.99).run(SHOW=False)
    # REINFORCE("soft", ENV_NAME="CartPole-v0", GAMMA=1, TEMPERATURE=1).run(SHOW=False)
    # REINFORCE("pyro", ENV_NAME="CartPole-v0", GAMMA=0.99, TEMPERATURE=1, PRIOR="unif", MODEL_MODE="plate").run(SHOW=False)
    # REINFORCE("pyro", ENV_NAME="CartPole-v0", GAMMA=0.99, TEMPERATURE=1, LEARNING_RATE=10e-4, PRIOR="unif", MODEL_MODE="sequential").run()
    # REINFORCE("pyro", ENV_NAME="CartPole-v0", GAMMA=1, TEMPERATURE=1, PRIOR="pi", MODEL_MODE="plate").run(SHOW=False)
    # REINFORCE("pyro", ENV_NAME="CartPole-v0", GAMMA=1, TEMPERATURE=1, PRIOR="pi", MODEL_MODE="sequential").run(SHOW=False)
