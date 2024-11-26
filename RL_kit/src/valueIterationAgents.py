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

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount=0.9, iterations=100):
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
        self.values = util.Counter()  # A Counter is a dict with default 0

        # Inicia um novo vetor de valores que deve guardar os novos valores
        newValues = util.Counter()

        # Gera um loop a partir do número de iterações informado
        for i in range(0, iterations):

            # Passa por cada estado existente
            for state in self.mdp.getStates():

                # Inicializa em -infinito a variável do maior Q-valor (pois podem haver valores negativos)
                maxQValue = float('-inf')

                # Se for estado terminal, o valor é 0
                if self.mdp.isTerminal(state):
                    maxQValue = 0
                #Se não for terminal, então pegamos o valor máximo que se pode obter a partir das ações possíveis
                else:
                    # Passa por cada ação possível
                    for action in self.mdp.getPossibleActions(state):

                        # Pega o Q-valor associado a ação no estado
                        qValue = self.computeQValueFromValues(state, action)

                        # Se o Q-valor atual for maior que os anteriores, então guardamos
                        if qValue > maxQValue:
                            maxQValue = qValue
                
                # Guarda o maior Q-Valor no novo vetor de valores
                newValues[state] = maxQValue
            
            # Guarda os novos valores no vetor antigo
            self.values = newValues.copy()

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
        # Vetor de valores
        values = []

        # Passa por cada par (nextState, prob) do retorno de self.mdp.getTransitionStatesAndProbs(state, action)
        for transitionStateAndProb in self.mdp.getTransitionStatesAndProbs(state, action):

            nextState = transitionStateAndProb[0]
            prob = transitionStateAndProb[1]

            # Adiciona ao vetor o resultado de P(s′|s,a)*γ*V(s′) onde s′ = nextState
            values.append(prob * ( self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState] ))
        
        # Retorna o somatório de P(s′|s,a)*γ*V(s′)
        return sum(values)


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # Coleta as possíveis ações a partir do estado dado
        possibleActions = self.mdp.getPossibleActions(state)

        # Verifica se estado é terminal. Se sim, retorna None.
        if self.mdp.isTerminal(state):
            return None
        
        # Inicializa a variável com a melhor ação
        bestAction = None

        #Inicializa em -infinito a variável do maior Q-valor (pois podem haver valores negativos)
        maxQValue = float('-inf')
        
        # Passa por cada possível ação da lista de retorno de self.mdp.getTransitionStatesAndProbs(state, action)
        for action in possibleActions:

            qValue = self.computeQValueFromValues(state, action)

            # Se o Q-Valor atual é maior que o maior Q-Valor até então, a mlehor ação até o momento é a atual
            if qValue > maxQValue:
                maxQValue = qValue
                bestAction = action
        
        # Retorna a melhor ação
        return bestAction


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
