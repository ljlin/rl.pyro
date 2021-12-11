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
import utils.buffers

import warnings

warnings.filterwarnings("ignore")

class AC(torch.nn.Module):
    """
    CS885 Fall 2021 - Reinforcement Learning
    https://cs.uwaterloo.ca/~ppoupart/teaching/cs885-fall21/schedule.html

    - MODE=hard: Actor Critic Algorithm
      Slides 9
      https://cs.uwaterloo.ca/~ppoupart/teaching/cs885-fall21/slides/cs885-lecture7b.pdf
    - Mode=soft: Soft Actor Critic (SAC) (pure PyTorch)
      Slides 22
      https://cs.uwaterloo.ca/~ppoupart/teaching/cs885-fall21/slides/cs885-module2.pdf
    - Mode=soft: Soft Actor Critic (SAC) (PyTorch + Pyro)
      Fellows, Matthew, Anuj Mahajan, Tim GJ Rudner, and Shimon Whiteson.
      "Virel: A variational inference framework for reinforcement learning."
      Advances in Neural Information Processing Systems 32 (2019): 7122-7136.
      https://arxiv.org/abs/1811.01132
    """

    def __init__(
            self,
            MODE,
            PRIOR=None,
            TEMPERATURE=None,
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
            EPISODES=300,
            # Total number of episodes to learn over
            TRAIN_AFTER_EPISODES=10,
            # Just collect episodes for these many episodes
            TRAIN_EPOCHS=25,
            # Train for these many epochs every time
            BUFSIZE=10000,
            # Replay buffer size
            TEST_EPISODES=1,
            # Test episodes after every train episode
            TARGET_UPDATE_FREQ=10,
            # Target network update frequency
            SVI_EPOCHS=None,
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
            self.TRAIN_AFTER_EPISODES = 10
            self.TARGET_UPDATE_FREQ = 5
        else:
            self.SEEDS = SEEDS
            self.EPISODES = EPISODES
            self.TRAIN_AFTER_EPISODES = TRAIN_AFTER_EPISODES
            self.TARGET_UPDATE_FREQ = TARGET_UPDATE_FREQ

        self.TRAIN_EPOCHS = TRAIN_EPOCHS
        self.BUFSIZE = BUFSIZE
        self.TEST_EPISODES = TEST_EPISODES

        assert (self.SOFT_OFF != self.SOFT_ON)
        assert (self.SOFT_ON == (TEMPERATURE is not None))
        self.TEMPERATURE = TEMPERATURE

        assert (self.SVI_ON == (PRIOR is not None))
        assert (self.SVI_ON == (SVI_EPOCHS is not None))
        if self.SVI_ON:
            assert(PRIOR is not None)
            self.PRIOR = PRIOR
            self.model = getattr(self, f"model_{PRIOR}", None)
            assert (self.model is not None)

            assert (SVI_EPOCHS is not None)
            self.SVI_EPOCHS = SVI_EPOCHS

        self.AC_MODE = ["QAC", "VAC", "DQAC"][-1]


    @property
    def SVI_ON(self):
        return self.MODE == "pyro"

    @property
    def SVI_OFF(self):
        res = self.MODE == "hard" or self.MODE == "soft"
        assert (self.SVI_ON != res)
        return res

    @property
    def SOFT_ON(self):
        return self.MODE == "pyro" or self.MODE == "soft"

    @property
    def SOFT_OFF(self):
        res = self.MODE == "hard"
        assert (self.SOFT_ON != res)
        return res

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
        self.unif = torch.ones(self.MINIBATCH_SIZE, self.ACT_N) / self.ACT_N

        self.log_pi = torch.nn.Sequential(
            torch.nn.Linear(self.OBS_N, self.HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(self.HIDDEN, self.HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(self.HIDDEN, self.ACT_N),
            # Clamp(max = 10, min = -10),
            # torch.nn.Softmax(dim=-1)
            torch.nn.LogSoftmax(dim=-1)
        ).to(self.DEVICE)

        buf = utils.buffers.ReplayBuffer(self.BUFSIZE)

        if self.SVI_ON:
            adma = pyro.optim.Adam({"lr": self.LEARNING_RATE})
            OPT_pi = pyro.infer.SVI(self.model, self.guide, adma, loss=pyro.infer.Trace_ELBO())
        else:
            OPT_pi = torch.optim.Adam(self.log_pi.parameters(), lr=self.LEARNING_RATE)

        if self.SOFT_ON or self.AC_MODE == "DQAC":
            self.Q = torch.nn.Sequential(
                torch.nn.Linear(self.OBS_N, self.HIDDEN), torch.nn.ReLU(),
                torch.nn.Linear(self.HIDDEN, self.HIDDEN), torch.nn.ReLU(),
                torch.nn.Linear(self.HIDDEN, self.ACT_N)
            ).to(self.DEVICE)
            self.Qt = torch.nn.Sequential(
                torch.nn.Linear(self.OBS_N, self.HIDDEN), torch.nn.ReLU(),
                torch.nn.Linear(self.HIDDEN, self.HIDDEN), torch.nn.ReLU(),
                torch.nn.Linear(self.HIDDEN, self.ACT_N)
            ).to(self.DEVICE)
            OPT = torch.optim.Adam(self.Q.parameters(), lr=self.LEARNING_RATE)
            return env, test_env, buf, self.pi, OPT_pi, self.Q, self.Qt, None, OPT
        else:
            if self.AC_MODE == "VAC":
                self.V = torch.nn.Sequential(
                    torch.nn.Linear(self.OBS_N, self.HIDDEN), torch.nn.ReLU(),
                    torch.nn.Linear(self.HIDDEN, self.HIDDEN), torch.nn.ReLU(),
                    torch.nn.Linear(self.HIDDEN, 1),
                ).to(self.DEVICE)
                OPT = torch.optim.Adam(self.V.parameters(), lr=self.LEARNING_RATE)
                return env, test_env, buf, self.pi, OPT_pi, None, None, self.V, OPT
            elif self.AC_MODE == "QAC":
                self.Q = torch.nn.Sequential(
                    torch.nn.Linear(self.OBS_N, self.HIDDEN), torch.nn.ReLU(),
                    torch.nn.Linear(self.HIDDEN, self.HIDDEN), torch.nn.ReLU(),
                    torch.nn.Linear(self.HIDDEN, self.ACT_N)
                ).to(self.DEVICE)
                OPT = torch.optim.Adam(self.Q.parameters(), lr=self.LEARNING_RATE)
                return env, test_env, buf, self.pi, OPT_pi, self.Q, None, None, OPT

    def pi(self,input):
        log_out = self.log_pi(input)
        return torch.exp(log_out)

    def guide(self, states):
        pyro.module("agent", self.log_pi)
        with pyro.plate("state_batch", states.shape[0], device=self.t.device):
            prob_action = self.pi(states)
            action = pyro.sample("action", pyro.distributions.Categorical(prob_action))

    def model_unif(self, states):
        with pyro.plate("state_batch", states.shape[0], device=self.t.device):
            action = pyro.sample("action", pyro.distributions.Categorical(self.unif))
            qvalues = self.Qt(states).detach().gather(1, action.view(-1, 1)).squeeze()
            pyro.factor("reward", qvalues / self.TEMPERATURE)

    def model_softmaxQ(self, states):
        with pyro.plate("state_batch", states.shape[0], device=self.t.device):
            prob_action = torch.nn.functional.softmax(
                self.Qt(states).detach() / self.TEMPERATURE,
                dim=-1
            )
            action = pyro.sample("action", pyro.distributions.Categorical(prob_action))

    # Update networks
    def update_networks(self, epi, buf, pi, OPT_Pi, Q, Qt, V, OPT_Q):
        # Sample a minibatch (s, a, r, s', d)
        # Each variable is a vector of corresponding values
        if self.SOFT_ON:
            S, A, R, S_prime, D = buf.sample(self.MINIBATCH_SIZE, self.t)
            A_prime = torch.distributions.Categorical(pi(S_prime)).sample()
            # Get Q(s, a) for every (s, a) in the minibatch
            qvalues = Q(S).gather(1, A.view(-1, 1)).squeeze()
            # Get Qt(s', a') for every (s') in the minibatch
            q_prime_values = Qt(S_prime).gather(1, A_prime.view(-1, 1)).squeeze()
        else:
            S, A, R, S_prime, D, N = buf.sample(self.MINIBATCH_SIZE, self.t)
            if self.AC_MODE == 'DQAC':
                A_prime = torch.distributions.Categorical(pi(S_prime)).sample()
                qvalues = Q(S).gather(1, A.view(-1, 1)).squeeze()
                q_prime_values = Qt(S_prime).gather(1, A_prime.view(-1, 1)).squeeze()

        # M Step
        if self.SVI_ON or self.AC_MODE == "DQAC":
            targets = R + self.GAMMA * q_prime_values * (1 - D)
            loss = torch.nn.functional.mse_loss(targets.detach(), qvalues)
        elif self.SOFT_ON:
            assert (self.MODE == 'soft')
            entropy = torch.mean(torch.distributions.Categorical(pi(S_prime)).entropy())
            targets = R + self.GAMMA * (q_prime_values + self.TEMPERATURE * entropy) * (1 - D)
            loss = torch.nn.functional.mse_loss(targets.detach(), qvalues)
        else:
            # delta = (R + self.GAMMA * Q(S_prime).max(-1)[0] - Q(S).gather(1, A.view(-1, 1)).squeeze()).detach()
            # loss = - (
            #     delta *
            #     torch.pow(self.GAMMA, N).detach() *
            #     Q(S).gather(1, A.view(-1, 1)).squeeze()
            # ).mean()

            # targets = R + self.GAMMA * q_prime_values * (1 - D)
            # loss = torch.nn.functional.smooth_l1_loss(qvalues, targets.detach())
            if self.AC_MODE == "VAC":
                delta = (R + self.GAMMA * V(S_prime) - V(S)).detach()
                loss = - (delta * V(S)).mean()
            elif self.AC_MODE == "QAC":
                A_prime = torch.distributions.Categorical(pi(S_prime)).sample()
                q_prime_values = Q(S_prime).gather(1, A_prime.view(-1, 1)).squeeze()
                qvalues = Q(S).gather(1, A.view(-1, 1)).squeeze()
                delta = (R + self.GAMMA * q_prime_values - qvalues).detach()
                loss = - (delta * qvalues).mean()

        OPT_Q.zero_grad()
        loss.backward()
        OPT_Q.step()

        # E Step
        if self.SVI_ON:
            for epoch in range(self.SVI_EPOCHS):
                OPT_Pi.step(S)
        else:
            if self.SOFT_ON:
                loss_policy = torch.nn.KLDivLoss(reduction='batchmean')(
                    torch.nn.functional.log_softmax(Qt(S).detach() / self.TEMPERATURE, dim=-1), pi(S)
                )
            else:
                if self.AC_MODE == "VAC":
                    loss_policy = - (
                        delta *
                        torch.pow(self.GAMMA, N) *
                        torch.log(pi(S).gather(-1, A.view(-1, 1))).squeeze()
                    ).mean()
                elif self.AC_MODE == "QAC":
                    # adv = (
                    #     Q(S).gather(1, A.view(-1, 1)).squeeze() -
                    #     (pi(S) * Q(S)).sum(-1)
                    # )
                    loss_policy = - (
                        qvalues.detach() *
                        torch.log(pi(S).gather(-1, A.view(-1, 1))).squeeze()
                    ).mean()
                elif self.AC_MODE == "DQAC":
                    adv = (
                        R + self.GAMMA * self.Qt(S_prime).max(-1)[0] - (pi(S) * Qt(S)).sum(-1) #slides
                        # (R + self.GAMMA * Qt(S_prime).gather(1, A_prime.view(-1, 1)).squeeze()) - Qt(S).gather(1, A.view(-1, 1)).squeeze() #?
                        # (R + self.GAMMA * (pi(S_prime) * Qt(S_prime)).sum(-1)) - Qt(S).gather(1, A.view(-1, 1)).squeeze()
                        # (R + self.GAMMA * (pi(S_prime) * Qt(S_prime)).sum(-1)) - (pi(S) * Qt(S)).sum(-1)
                    ).detach()
                    loss_policy = - (
                        adv *
                        # torch.pow(self.GAMMA, N).detach() *
                        self.log_pi(S).gather(-1, A.view(-1, 1)).squeeze()
                    ).mean()

            OPT_Pi.zero_grad()
            loss_policy.backward()
            OPT_Pi.step()
        # print(loss_policy.item())
        # Update target network every few steps
        if (self.SOFT_ON  or self.AC_MODE == "DQAC" ) and epi % self.TARGET_UPDATE_FREQ == 0:
            utils.common.update(Qt, Q)

        return loss.item()

    def train(self, seed):

        print("Seed=%d" % seed)
        env, test_env, buf, pi, OPT_pi, Q, Qt, V, OPT = self.create_everything(seed)

        if self.SVI_ON:
            pyro.clear_param_store()

        def policy(env, obs):
            with torch.no_grad():
                obs = self.t.f(obs).view(-1, self.OBS_N)  # Convert to torch tensor
                action = torch.distributions.Categorical(pi(obs)).sample().item()
            return action

        testRs = []
        last25testRs = []
        print("Training:")
        pbar = tqdm.trange(self.EPISODES)
        for epi in pbar:

            # Play an episode and log episodic reward
            if self.SOFT_ON:
                S, A, R = utils.envs.play_episode_rb(env, policy, buf)
            else:
                S, A, R = utils.envs.play_episode_rb_with_steps(env,policy,buf)

            # Train after collecting sufficient experience
            if epi >= self.TRAIN_AFTER_EPISODES:

                # Train for TRAIN_EPOCHS
                for tri in range(self.TRAIN_EPOCHS):
                    self.update_networks(epi, buf, pi, OPT_pi, Q, Qt, V, OPT)

            # Evaluate for TEST_EPISODES number of episodes
            Rews = []
            for epj in range(self.TEST_EPISODES):
                S, A, R = utils.envs.play_episode(test_env, policy, render=False)
                Rews += [sum(R)]
            testRs += [sum(Rews) / self.TEST_EPISODES]

            # Update progress bar
            last25testRs += [sum(testRs[-25:]) / len(testRs[-25:])]
            pbar.set_description("R25(%g)" % (last25testRs[-1]))
        # Close progress bar, environment
        pbar.close()
        print("Training finished!")
        env.close()
        test_env.close()

        return last25testRs

    def run(self, info=None, SHOW = True):
        # Train for different seeds
        label=f"AC-{self.MODE}"
        if self.SVI_ON:
            label += f"-prior_{self.PRIOR}"
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
    AC(
        "hard",
        # SMOKE_TEST=True,
        # LEARNING_RATE=5e-5,
        SEEDS=[1],
        # EPISODES=3000
    ).run("DQAC-GAMMA")

    # AC(
    #     "pyro",
    #     # SMOKE_TEST=True,
    #     PRIOR="unif",
    #     TEMPERATURE=1,
    #     SVI_EPOCHS=1,
    #     SEEDS=[1],
    #     EPISODES=300
    # ).run()
