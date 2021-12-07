from unittest import TestCase
import REINFORCE
import AC

class Test(TestCase):
    def test_reinforce(self):
        REINFORCE.REINFORCE("hard", SMOKE_TEST=True)
        REINFORCE.REINFORCE("soft", SMOKE_TEST=True, TEMPERATURE=1)
        REINFORCE.REINFORCE("pyro", SMOKE_TEST=True, TEMPERATURE=1, UNIF_PRIOR=True)
        REINFORCE.REINFORCE("pyro", SMOKE_TEST=True, TEMPERATURE=1, UNIF_PRIOR=False)
    def test_ac(self):
        AC.AC("hard", SMOKE_TEST=True)
        AC.AC("soft", SMOKE_TEST=True, TEMPERATURE=1)
        AC.AC("pyro", SMOKE_TEST=True, TEMPERATURE=1, PRIOR="unif", SVI_EPOCHS = 1)
        AC.AC("pyro", SMOKE_TEST=True, TEMPERATURE=1, PRIOR="softmaxQ", SVI_EPOCHS = 1)