from unittest import TestCase
import REINFORCE
import AC

class Test(TestCase):
    def test_reinforce(self):
        REINFORCE.REINFORCE("hard", ENV_NAME="CartPole-v0", GAMMA = 1, SMOKE_TEST=True).run(SHOW=False)
        REINFORCE.REINFORCE("soft", ENV_NAME="CartPole-v0", GAMMA = 1, SMOKE_TEST=True, TEMPERATURE=1).run(SHOW=False)
        REINFORCE.REINFORCE("pyro", ENV_NAME="CartPole-v0", GAMMA = 1, SMOKE_TEST=True, TEMPERATURE=1, PRIOR="unif", MODEL_MODE="plate").run(SHOW=False)
        REINFORCE.REINFORCE("pyro", ENV_NAME="CartPole-v0", GAMMA = 1, SMOKE_TEST=True, TEMPERATURE=1, PRIOR="unif", MODEL_MODE="sequential").run(SHOW=False)
        REINFORCE.REINFORCE("pyro", ENV_NAME="CartPole-v0", GAMMA = 1, SMOKE_TEST=True, TEMPERATURE=1, PRIOR="pi", MODEL_MODE="plate").run(SHOW=False)
        REINFORCE.REINFORCE("pyro", ENV_NAME="CartPole-v0", GAMMA = 1, SMOKE_TEST=True, TEMPERATURE=1, PRIOR="pi", MODEL_MODE="sequential").run(SHOW=False)
    def test_ac(self):
        AC.AC("hard", ENV_NAME="CartPole-v0", GAMMA = 1, SMOKE_TEST=True).run(SHOW=False)
        AC.AC("soft", ENV_NAME="CartPole-v0", GAMMA = 1, SMOKE_TEST=True, TEMPERATURE=1).run(SHOW=False)
        AC.AC("pyro", ENV_NAME="CartPole-v0", GAMMA = 1, SMOKE_TEST=True, TEMPERATURE=1, PRIOR="unif", SVI_EPOCHS = 1).run(SHOW=False)
        AC.AC("pyro", ENV_NAME="CartPole-v0", GAMMA = 1, SMOKE_TEST=True, TEMPERATURE=1, PRIOR="softmaxQ", SVI_EPOCHS = 1).run(SHOW=False)