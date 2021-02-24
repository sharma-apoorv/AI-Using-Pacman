# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

import copy
import operator

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

        return

    def runValueIteration(self):
        
        states_list = self.mdp.getStates()
        values = util.Counter()

        for _ in range(self.iterations):
            for state in states_list:
                actions_list = self.mdp.getPossibleActions(state)
                for action in actions_list:
                    values[(state, action)] = self.computeQValueFromValues(state, action)
                
                # if not self.mdp.isTerminal(state):
                #     self.values[state] = max([values[(state, action)] for action in actions_list])
            
            for state in states_list:
                actions_list = self.mdp.getPossibleActions(state)
                if not self.mdp.isTerminal(state):
                    self.values[state] = max([values[(state, action)] for action in actions_list])

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """

        q_value = 0
        
        transition_tuples = self.mdp.getTransitionStatesAndProbs(state, action)
        for next_state, p in transition_tuples:
            transition_reward = self.mdp.getReward(state, action, next_state)
            q_value += p * (transition_reward + self.discount * self.values[next_state])
        
        return q_value


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        
        if self.mdp.isTerminal(state):
            return None
        
        q_values = {}
        actions_list = self.mdp.getPossibleActions(state)
        for action in actions_list:
            q_values[action] = self.computeQValueFromValues(state, action)
        
        return max(q_values.items(), key=operator.itemgetter(1))[0]


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states_list = self.mdp.getStates()
        num_states = len(states_list)
        values = util.Counter()

        for i in range(self.iterations):
            state_idx = i % num_states
            state = states_list[state_idx]

            if self.mdp.isTerminal(state): 
                continue

            actions_list = self.mdp.getPossibleActions(state)
            for action in actions_list:
                values[(state, action)] = self.computeQValueFromValues(state, action)
            
            self.values[state] = max([values[(state, action)] for action in actions_list])
                    

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):

        self.pq = util.PriorityQueue()
        self.predecessors = {}
        
        states_list = self.mdp.getStates()

        self.compute_predecessors(states_list)
        self.populate_priority_queue(states_list)
        self.compute_iterations(states_list)
    
    def compute_predecessors(self, states_list):

        for curr_state in states_list:
            prev_states = set()

            for prev_state in states_list:
                
                actions_list = self.mdp.getPossibleActions(prev_state)
                for action in actions_list:

                    transition_tuples = self.mdp.getTransitionStatesAndProbs(prev_state, action)
                    for next_state, p in transition_tuples:

                        # if there is a probability and we can get from the predecessor
                        # to the current state we are on ... 
                        if (p > 0) and (next_state == curr_state):
                            prev_states.add(prev_state)
            
            self.predecessors[curr_state] = prev_states

    def populate_priority_queue(self, states_list):

        for state in states_list:
            if self.mdp.isTerminal(state): 
                continue
                
            curr_state_value = self.getValue(state)
            actions_list = self.mdp.getPossibleActions(state)
            values = util.Counter()
            for action in actions_list:
                values[(state, action)] = self.computeQValueFromValues(state, action)
            
            max_value = max([values[(state, action)] for action in actions_list])
            diff = abs(max_value - curr_state_value)

            # -We use a negative because the priority queue is a min heap, 
            # but we want to prioritize updating states that have a higher error
            self.pq.push(state, -diff)
    
    def compute_iterations(self, states_list):

        for i in range(self.iterations):

            if self.pq.isEmpty(): #priority queue is empty, then terminate
                break
                
            state = self.pq.pop()
            if not self.mdp.isTerminal(state):
            
                q_values = []
                actions_list = self.mdp.getPossibleActions(state)
                for action in actions_list:
                    
                    q_value = 0
                    transition_tuples = self.mdp.getTransitionStatesAndProbs(state, action)
                    for next_state, p in transition_tuples:
                        transition_reward = self.mdp.getReward(state, action, next_state)
                        q_value += p * (transition_reward + self.discount * self.values[next_state])

                    q_values.append(q_value)
                
                self.values[state] = max(q_values)
            
            for prev_state in self.predecessors[state]:
                
                current = self.values[prev_state]
                
                q_values = []
                actions_list = self.mdp.getPossibleActions(prev_state)
                for action in self.mdp.getPossibleActions(prev_state):
                    q_values.append(self.computeQValueFromValues(prev_state, action))
                
                max_value = max(q_values)
                diff = abs((current - max_value))
                
                if diff > self.theta:
                    self.pq.update(prev_state, -diff)

