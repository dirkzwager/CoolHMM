from math import log


class HMM:
    """
    HMM or a Hidden Markov Model is a class that represents a Bayesian network that models a Markov chain.
    A HMM allows us to 'reveal' a series of 'hidden' states that best fit a series of observed events.

    This class allows us to be a very Cool fortune teller indeed!
    """

    # A collection of states { S1, ...., Sn }
    S = ()

    # 2D hashtable of transition probabilities between states where aij = p(qt = Sj | q(t-1) = Si) where aij is the
    # transition between states from state Si to state Sj, qt the 'revealed' state at t or t-1
    A = {}

    # 2D hashtable of emission probabilities for observations associated with state Si where
    # bi(k) = p(ot = xk | qt = Si) for ot is observation at time t, xk is result xk, qt is 'revealed' state at time t.
    B = {}

    # hashtable of initial state distribution
    p = {}

    def __init__(self, states, emissions, transitions, initial_probability):
        self.S = states
        self.B = emissions
        self.A = transitions
        self.p = initial_probability

    def print_model(self):
        print('<========================================[ HMM ]========================================>')
        print('States:{}'.format(self.S))
        print('Emission probablities:{}'.format(self.B))
        print('Transition probabilities:{}'.format(self.A))
        print('Initial distribution:{}'.format(self.p))
        print('<=======================================================================================>')

    def viterbi(self, O):
        """The viterbi algorithm is used to find the optimal sequence of (hidden) states that explain a sequence of
        observed events by applying the Bellman Principle on a Hidden Markov Model.

        Argument:
        O -- array of observations ['O1', 'O2', .., 'On']

        Result:
        viterbi_path -- an array of tuples [(Ot, (St, pt)), ...] where Ot is the observation a time t, St the revealed
        state at time t and pt the probability at time t.
        """
        b = []  # array of backpointers
        f = {state: self.p[state] * self.B[state][O[0]] for state in self.S}  # compute the initial forward probability

        # for the remaining observations
        for t, o in enumerate(O[1:]):
            f_o = {state: float('-inf') for state in self.S}
            b_o = {state: () for state in self.S}

            # now loop over all combinations of states to find the transition with the highest forward probability
            for s2 in self.S:
                for s1 in self.S:
                    # compute the forward probability for transition s2 to s2 given the emission prob for being in
                    # s2 while observing o
                    f_ij = (f[s1] * self.A[s1][s2]) * self.B[s2][o]
                    if f_ij > f_o[s2]:
                        f_o[s2] = f_ij
                        b_o[s2] = (s1, f_ij)
            f = f_o  # propagate the forward probability to next observation
            b.append(b_o)  # leave behind a 'breadcrumb' to backtrack our path home

        # argmax to find the first backpointer
        viterbi_path = [max(f.items(), key=lambda x: x[1])]

        # now reverse over backpointers to obtain the final path
        for o in reversed(b):
            viterbi_path.insert(0, o[viterbi_path[0][0]])

        return viterbi_path

    def viterbi_logarithm(self, O):
        """The viterbi_logarithm function is used to find the optimal sequence of (hidden) states that explain a sequence
        of observed events by applying the Bellman Principle on a Hidden Markov Model. The logarithm with
        base 2 is used to avoid floating point errors due to multiplication of probability chains which can result in
        very small probabilities. In addition, by calculating with the log over the probabilities we are allowed
        to calculate with sums instead of products which provides more efficiency and accuracy.

        Argument:
        O -- array of observations ['O1', 'O2', .., 'On']

        Result:
        viterbi_path -- an array of tuples [(Ot, (St, pt)), ...] where Ot is the observation a time t, St the 'revealed'
        state at time t and pt the exponent over the base of our logarithmic function (2) at time t.

        Note: To get the probability of a certain observation Ot, one only has to compute 2 to the power of pt
        (the base of the logarithm to the power of pt).
        """
        b = []  # array of backpointers
        f = {state: -log(self.p[state], 2) - log(self.B[state][O[0]], 2) for state in
             self.S}  # compute the initial forward probability

        # for the remaining observations
        for t, o in enumerate(O[1:]):
            f_o = {state: float('inf') for state in self.S}
            b_o = {state: () for state in self.S}

            # now loop over all combination of states to find the transition with the highest forward probability
            for s2 in self.S:
                for s1 in self.S:
                    # compute the forward probability for transition s2 to s2 given the emission probability for being
                    # in s2 while observing o
                    f_ij = (f[s1] - log(self.A[s1][s2], 2)) - log(self.B[s2][o], 2)
                    if f_ij < f_o[s2]:
                        f_o[s2] = f_ij
                        b_o[s2] = (s1, f_ij)
            f = f_o  # propagate the forward probability to next observation
            b.append(b_o)  # leave behind a 'breadcrumb' to backtrack our path home

        # argmin to find the first backpointer
        viterbi_path = [min(f.items(), key=lambda x: x[1])]

        # now reverse over backpointers to obtain the final path
        for o in reversed(b):
            viterbi_path.insert(0, o[viterbi_path[0][0]])

        return viterbi_path


def print_path(observations, hidden_states):
    for i, o in enumerate(observations):
        print('O:{: >20}\t\tS:{: >20}'.format(o, hidden_states[i]))


def CB_demo():
    # Tuple of all possible states.
    states = ('Female', 'Male')

    # Dictionary assuming an equal distribution of initial states. In other words, it is with equal probability
    # that a session is initiated by either gender.
    start_probability = {'Female': 0.50, 'Male': 0.50}

    # 2D hashtable specifying the likelihood of transition between states. This encapsulates the scenario where
    # during a browsing session there is a switch between genders.
    transition_probability = {
        'Female': {'Female': 0.90, 'Male': 0.10},
        'Male': {'Female': 0.10, 'Male': 0.90}
    }

    # 2D hashtable of emission probabilities which specifies the distribution of an observation
    # (viewing of a certain item in the catalog) given our hidden variable is in a certain state (male or female).
    emission_probability = {
        'Female': {'mobile phone': 0.25, 'hairdryer': 0.31, 'parfum': 0.37, 'bbq': 0.07, 'weatherstation': 0.01},
        'Male': {'mobile phone': 0.25, 'hairdryer': 0.01, 'parfum': 0.20, 'bbq': 0.20, 'weatherstation': 0.34}
    }

    hmm = HMM(states, emission_probability, transition_probability, start_probability)
    hmm.print_model()

    print("\nOutput browsing session 1:")
    session1 = ['mobile phone', 'bbq', 'parfum', 'parfum', 'weatherstation']
    path = hmm.viterbi(session1)
    print_path(session1, path)

    print("\nOutput browsing session 1 using logarithm:")
    path = hmm.viterbi_logarithm(session1)
    print_path(session1, path)

    print("\nOutput browsing session 2:")
    session2 = ['hairdryer', 'parfum', 'mobile phone', 'weatherstation']
    path = hmm.viterbi(session2)
    print_path(session2, path)

    print("\nOutput browsing session 2 using logarithm:")
    path = hmm.viterbi_logarithm(session2)
    print_path(session2, path)

    print("\nOutput browsing session 3:")
    session3 = ['hairdryer', 'parfum', 'bbq', 'bbq', 'hairdryer', 'weatherstation', 'hairdryer']
    path = hmm.viterbi(session3)
    print_path(session3, path)

    print("\nOutput browsing session 3 using logarithm:")
    path = hmm.viterbi_logarithm(session3)
    print_path(session3, path)


CB_demo()

# <========================================[ HMM ]========================================>
# States:('Female', 'Male')
# Emission probablities:{'Male': {'weatherstation': 0.34, 'hairdryer': 0.01, 'parfum': 0.2, 'mobile phone': 0.25,
# 'bbq': 0.2}, 'Female': {'weatherstation': 0.01, 'hairdryer': 0.31, 'parfum': 0.37, 'mobile phone': 0.25, 'bbq': 0.07}}
# Transition probabilities:{'Male': {'Male': 0.9, 'Female': 0.1}, 'Female': {'Male': 0.1, 'Female': 0.9}}
# Initial distribution:{'Male': 0.5, 'Female': 0.5}
# <=======================================================================================>

# Output browsing session 1:
# O:        mobile phone          S:('Male', 0.022500000000000003)
# O:                 bbq          S:('Male', 0.004050000000000001)
# O:              parfum          S:('Male', 0.0007290000000000002)
# O:              parfum          S:('Male', 0.00022307400000000008)
# O:      weatherstation          S:('Male', 0.00022307400000000008)

# Output browsing session 1 using logarithm:
# O:        mobile phone          S:('Male', 5.473931188332412)
# O:                 bbq          S:('Male', 7.947862376664824)
# O:              parfum          S:('Male', 10.421793564997236)
# O:              parfum          S:('Male', 12.13019000696667)
# O:      weatherstation          S:('Male', 12.13019000696667)

# Output browsing session 2:
# O:           hairdryer          S:('Female', 0.051615)
# O:              parfum          S:('Female', 0.011613375)
# O:        mobile phone          S:('Female', 0.0003948547500000001)
# O:      weatherstation          S:('Male', 0.0003948547500000001)

# Output browsing session 2 using logarithm:
# O:           hairdryer          S:('Female', 4.276065796978674)
# O:              parfum          S:('Female', 6.428068890423724)
# O:        mobile phone          S:('Female', 11.30639033383547)
# O:      weatherstation          S:('Male', 11.30639033383547)

# Output browsing session 3:
# O:           hairdryer          S:('Female', 0.051615)
# O:              parfum          S:('Female', 0.0032517450000000003)
# O:                 bbq          S:('Female', 0.00020485993500000006)
# O:                 bbq          S:('Female', 5.715592186500002e-05)
# O:           hairdryer          S:('Female', 5.144032967850002e-07)
# O:      weatherstation          S:('Female', 1.4351851980301506e-07)
# O:           hairdryer          S:('Female', 1.4351851980301506e-07)

# Output browsing session 3 using logarithm:
# O:           hairdryer          S:('Female', 4.276065796978674)
# O:              parfum          S:('Female', 8.264570158140845)
# O:                 bbq          S:('Female', 12.253074519303016)
# O:                 bbq          S:('Female', 14.094737492135916)
# O:           hairdryer          S:('Female', 20.89059677535569)
# O:      weatherstation          S:('Female', 22.73225974818859)
# O:           hairdryer          S:('Female', 22.73225974818859)
