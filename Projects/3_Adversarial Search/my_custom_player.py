
from math import sqrt
from sample_players import DataPlayer
from isolation import Isolation, DebugState
import random
import numpy as np


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        
        # start by putting a random action in the queue
        action = random.choice(state.actions())
        self.queue.put(action)
        baseline_agent = False # baseline agent is alpha beta pruning agent. When False, MCTS agent is used
        if baseline_agent:
            self.init_alpha_beta(state)
        else:
            self.init_monte_carlo_tree_search(state)
        while True:
            if baseline_agent:
                action, val = self.alpha_beta_step()
            else:
                action, val = self.monte_carlo_tree_search_step()
            self.queue.put(action)
    
    # find actions in the lower right quadrant of the board for the opening move. 
    # Given the board symmetry, we only need to consider opening actions in one quadrant of the board.
    def lower_right_quad_actions(self, state):
        width = 0
        height = 0
        for i in range(2, len(state.actions())):
            x, y = DebugState.ind2xy(i)
            if x == 0:
                width = i-2
                height = len(state.actions()) / width
                break
        actions = []
        for i in state.actions():
            x, y = DebugState.ind2xy(i)
            if x <= (width + 1) // 2 and y <= (height+1) // 2: 
                actions.append(i)
        return actions

##  alpha beta pruning tree search with incremental deepening

    def init_alpha_beta(self, state):
        self.alpha_beta_depth = 0
        self.alpha_beta_state = state
        if state.ply_count == 0:
            actions = self.lower_right_quad_actions(state)
        else:
            actions = state.actions()
        self.alpha_beta_actions = actions
        self.consistent_best_action_count = 0
        self.prev_best_action = random.choice(actions)
    
    def alpha_beta_step(self):
        # with every call to the alpha_beta_step we increase the tree depth (incremental deepening)
        self.alpha_beta_depth += 1
        best_action, max_val = self.alpha_beta(self.alpha_beta_state, self.alpha_beta_actions, self.alpha_beta_depth)
        self.prev_best_action = best_action
        return best_action, max_val

    def alpha_beta(self, state, actions, depth):
        max_val = float("-inf")
        best_actions = []
        for action in actions:
            value = self.min_val(state.result(action), depth - 1)
            if value > max_val:
                best_actions = [action]
                max_val = value
            elif value == max_val:
                best_actions.append(action)
        return random.choice(best_actions), max_val
    
    def min_val(self, state, depth, lower_bound=float("-inf")):
        if state.terminal_test(): return state.utility(self.player_id)
        if depth <= 0: return self.score(state)
        value = float("inf")
        for action in state.actions():
            value = min(value, self.max_val(state.result(action), depth - 1, value))
            # if value is lower than the lower bound for this node, the rest of actions in the node can ignored (pruned)
            if value < lower_bound:
                return value
        return value

    def max_val(self, state, depth, upper_bound=float("inf")):
        if state.terminal_test(): return state.utility(self.player_id)
        if depth <= 0: return self.score(state)
        value = float("-inf")
        for action in state.actions():
            value = max(value, self.min_val(state.result(action), depth - 1, value))
            # if value is higher than the upper bound for this node, the rest of actions in the node can ignored (pruned)
            if value > upper_bound:
                return value
        return value

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)


##  Monter Carlo Tree Search


    def init_monte_carlo_tree_search(self, root):
        self.tree = {}
        self.root = root
        if root.ply_count == 0:
            actions = self.lower_right_quad_actions(root)
        else:
            actions = root.actions()
        # since argmax returns the first item when there are multiple items with the same max value, I shuffled the actions so that 
        # the order of action selection is randomized between runs. 
        random.shuffle(actions)
        # create the root node
        self.tree[root] = {'depth': 0,
                           'state_visits': 0, 
                           'actions': actions, 
                           'actions_visits': np.zeros(len(actions)), 
                           'actions_wins': np.zeros(len(actions))}

    def monte_carlo_tree_search_step(self):
        # number of MCTS iterations per step. 
        updates_per_call = 100 
        for i in range(updates_per_call):
            utility = self.expand_node(self.root)
        root_dict = self.tree[self.root]
        best_action_idx = np.argmax(root_dict['actions_wins'] / root_dict['actions_visits'])
        return root_dict['actions'][best_action_idx], root_dict['actions_wins'][best_action_idx] / root_dict['actions_visits'][best_action_idx]

    def expand_node(self, state):
        # the `state_visits` is increased here so that the `ln(state_visits)` be meaningful in the first call, the impact on the accuracy of UCB1 function should be minimal
        self.tree[state]['state_visits'] += 1
        # select an action based on UCB1 equation
        action_idx = self.select_action(self.tree[state])
        # expand the state given the selected action
        utility = self.expand_action(state, self.tree[state]['actions'][action_idx])
        self.tree[state]['actions_visits'][action_idx] += 1
        # if the game is won and max level, increase the win count
        if utility > 0 and self.tree[state]['depth'] % 2 == 0:
            self.tree[state]['actions_wins'][action_idx] += 1
        # if the game is lost and min level, increase the win count (for the opponent)
        elif utility < 0 and self.tree[state]['depth'] % 2 == 1:
            self.tree[state]['actions_wins'][action_idx] += 1
        return utility

    def select_action(self, state_dict):
        # adding a small number to avoid division by zero for actions that have not been expanded yet. 
        actions_visits = state_dict['actions_visits'] + 1e-3
        c = 1
        ucb = state_dict['actions_wins']/actions_visits + c * np.sqrt(2 * np.log(state_dict['state_visits']) / actions_visits)
        return np.argmax(ucb)

    def expand_action(self, state, action):
        next_state = state.result(action)
        # if terminal state return the utility
        if next_state.terminal_test():
            return next_state.utility(self.player_id)
        # if leaf node (depth is too high, or too many nodes expanded) the do monte carlo simulation
        if self.tree[state]['depth'] + 1 >= 6 and len(self.tree) > 200:
            return self.monte_carlo_sim(next_state)
        # otherwise expand the node
        
        # if node is not present in the tree, add to the tree dictionary
        if next_state not in self.tree:
            actions = next_state.actions()
            random.shuffle(actions)
            self.tree[next_state] = {'depth': self.tree[state]['depth'] + 1, 
                                     'state_visits': 0, 
                                     'actions': actions, 
                                     'actions_visits': np.zeros(len(actions)), 
                                     'actions_wins': np.zeros(len(actions))}
        return self.expand_node(next_state)

    def monte_carlo_sim(self, state):
        # take a random uniform action from possible actions until we reach a terminal state. 
        if state.terminal_test(): 
            return state.utility(self.player_id)
        else:
            return self.monte_carlo_sim(state.result(random.choice(state.actions())))

