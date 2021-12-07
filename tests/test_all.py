from unittest import TestCase
import REINFORCE
import AC

class Test(TestCase):
    def test_reinforce(self):
        REINFORCE.REINFORCE("hard", SMOKE_TEST=True).run(SHOW=False)
        REINFORCE.REINFORCE("soft", SMOKE_TEST=True, TEMPERATURE=1).run(SHOW=False)
        REINFORCE.REINFORCE("pyro", SMOKE_TEST=True, TEMPERATURE=1, UNIF_PRIOR=True).run(SHOW=False)
        REINFORCE.REINFORCE("pyro", SMOKE_TEST=True, TEMPERATURE=1, UNIF_PRIOR=False).run(SHOW=False)
    def test_ac(self):
        AC.AC("hard", SMOKE_TEST=True).run(SHOW=False)
        AC.AC("soft", SMOKE_TEST=True, TEMPERATURE=1).run(SHOW=False)
        AC.AC("pyro", SMOKE_TEST=True, TEMPERATURE=1, PRIOR="unif", SVI_EPOCHS = 1).run(SHOW=False)
        AC.AC("pyro", SMOKE_TEST=True, TEMPERATURE=1, PRIOR="softmaxQ", SVI_EPOCHS = 1).run(SHOW=False)