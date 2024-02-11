import numpy as np

import reversi
import time


class ReversiBot:
    def __init__(self, move_num):
        self.move_num = move_num

    def make_move(self, state):
        '''
        This is the only function that needs to be implemented for the lab!
        The bot should take a game state and return a move.

        The parameter "state" is of type ReversiGameState and has two useful
        member variables. The first is "board", which is an 8x8 numpy array
        of 0s, 1s, and 2s. If a spot has a 0 that means it is unoccupied. If
        there is a 1 that means the spot has one of player 1's stones. If
        there is a 2 on the spot that means that spot has one of player 2's
        stones. The other useful member variable is "turn", which is 1 if it's
        player 1's turn and 2 if it's player 2's turn.

        ReversiGameState objects have a nice method called get_valid_moves.
        When you invoke it on a ReversiGameState object a list of valid
        moves for that state is returned in the form of a list of tuples.

        Move should be a tuple (row, col) of the move you want the bot to make.
        '''
        # valid_moves = state.get_valid_moves()

        # move = rand.choice(valid_moves) # Moves randomly...for now
        # return move
        start_time = time.time()
        best_score, best_move = self.alphaBeta(state, depth=3)
        end_time = time.time()
        print(f'time to play: {end_time - start_time}')
        return best_move

    def minimax(self, state, depth, maximizing_player=True):
        if depth == 0 or len(state.get_valid_moves()) == 0:
            return self.evaluate(state), None

        valid_moves = state.get_valid_moves()

        if maximizing_player:
            best_score = float('-inf')
            best_move = None
            for move in valid_moves:
                new_state = reversi.ReversiGameState(np.copy(state.board), state.turn)
                new_state.simulate_move(move[0], move[1])
                score, _ = self.minimax(new_state, depth - 1, False)
                if score > best_score:
                    best_score = score
                    best_move = move
            return best_score, best_move
        else:
            best_score = float('inf')
            best_move = None
            for move in valid_moves:
                new_state = reversi.ReversiGameState(np.copy(state.board), state.turn)
                new_state.simulate_move(move[0], move[1])
                score, _ = self.minimax(new_state, depth - 1, True)
                if score < best_score:
                    best_score = score
                    best_move = move
            return best_score, best_move

    def alphaBeta(self, state, depth, alpha=float('-inf'), beta=float('inf'), maximizing_player=True):
        if depth == 0 or len(state.get_valid_moves()) == 0:
            return self.evaluate(state), None

        valid_moves = state.get_valid_moves()

        if maximizing_player:
            best_score = float('-inf')
            best_move = None
            for move in valid_moves:
                new_state = reversi.ReversiGameState(np.copy(state.board), state.turn)
                new_state.simulate_move(move[0], move[1])
                score, _ = self.alphaBeta(new_state, depth - 1, alpha, beta, False)
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return best_score, best_move
        else:
            best_score = float('inf')
            best_move = None
            for move in valid_moves:
                new_state = reversi.ReversiGameState(np.copy(state.board), state.turn)
                new_state.simulate_move(move[0], move[1])
                score, _ = self.alphaBeta(new_state, depth - 1, alpha, beta, True)
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)
                if alpha <= beta:
                    break
            return best_score, best_move

    def evaluate(self, state):
    # this function came from the github link on this website: https://kartikkukreja.wordpress.com/2013/03/30/heuristic-function-for-reversiothello/
    # (converted from C++ to python)
    # for now let's assume that me(the bot) is player 2 and opponent(human) is player 1
        my_color = 2
        opp_color = 1

        my_tiles = 0
        opp_tiles = 0

        my_front_tiles = 0
        opp_front_tiles = 0

        X1 = [-1, -1, 0, 1, 1, 1, 0, -1]
        Y1 = [0, 1, 1, 1, 0, -1, -1, -1]
        V = [
            [20, -3, 11, 8, 8, 11, -3, 20],
            [-3, -7, -4, 1, 1, -4, -7, -3],
            [11, -4, 2, 2, 2, 2, -4, 11],
            [8, 1, 2, -3, -3, 2, 1, 8],
            [8, 1, 2, -3, -3, 2, 1, 8],
            [11, -4, 2, 2, 2, 2, -4, 11],
            [-3, -7, -4, 1, 1, -4, -7, -3],
            [20, -3, 11, 8, 8, 11, -3, 20]
        ]

        # Coin Parity
        d = 0
        for i in range(8):
            for j in range(8):
                if state.board[i][j] == my_color:
                    d += V[i][j]
                    my_tiles += 1
                elif state.board[i][j] == opp_color:
                    d -= V[i][j]
                    opp_tiles += 1
                if state.board[i][j] != 0:
                    for k in range(8):
                        x = i + X1[k]
                        y = j + Y1[k]
                        if 0 <= x < 8 and 0 <= y < 8 and state.board[x][y] == 0:
                            if state.board[i][j] == my_color:
                                my_front_tiles += 1
                            else:
                                opp_front_tiles += 1
                            break

        if my_tiles > opp_tiles:
            p = (100.0 * my_tiles) / (my_tiles + opp_tiles)
        elif my_tiles < opp_tiles:
            p = -(100.0 * opp_tiles) / (my_tiles + opp_tiles)
        else:
            p = 0

        if my_front_tiles > opp_front_tiles:
            f = -(100.0 * my_front_tiles) / (my_front_tiles + opp_front_tiles)
        elif my_front_tiles < opp_front_tiles:
            f = (100.0 * opp_front_tiles) / (my_front_tiles + opp_front_tiles)
        else:
            f = 0

        # Corner occupancy
        my_tiles = opp_tiles = 0
        if state.board[0][0] == my_color:
            my_tiles += 1
        elif state.board[0][0] == opp_color:
            opp_tiles += 1
        if state.board[0][7] == my_color:
            my_tiles += 1
        elif state.board[0][7] == opp_color:
            opp_tiles += 1
        if state.board[7][0] == my_color:
            my_tiles += 1
        elif state.board[7][0] == opp_color:
            opp_tiles += 1
        if state.board[7][7] == my_color:
            my_tiles += 1
        elif state.board[7][7] == opp_color:
            opp_tiles += 1
        c = 25 * (my_tiles - opp_tiles)

        # Corner closeness
        my_tiles = opp_tiles = 0
        if state.board[0][0] == '-':
            if state.board[0][1] == my_color:
                my_tiles += 1
            elif state.board[0][1] == opp_color:
                opp_tiles += 1
            if state.board[1][1] == my_color:
                my_tiles += 1
            elif state.board[1][1] == opp_color:
                opp_tiles += 1
            if state.board[1][0] == my_color:
                my_tiles += 1
            elif state.board[1][0] == opp_color:
                opp_tiles += 1
        if state.board[0][7] == '-':
            if state.board[0][6] == my_color:
                my_tiles += 1
            elif state.board[0][6] == opp_color:
                opp_tiles += 1
            if state.board[1][6] == my_color:
                my_tiles += 1
            elif state.board[1][6] == opp_color:
                opp_tiles += 1
            if state.board[1][7] == my_color:
                my_tiles += 1
            elif state.board[1][7] == opp_color:
                opp_tiles += 1
        if state.board[7][0] == '-':
            if state.board[7][1] == my_color:
                my_tiles += 1
            elif state.board[7][1] == opp_color:
                opp_tiles += 1
            if state.board[6][1] == my_color:
                my_tiles += 1
            elif state.board[6][1] == opp_color:
                opp_tiles += 1
            if state.board[6][0] == my_color:
                my_tiles += 1
            elif state.board[6][0] == opp_color:
                opp_tiles += 1
        if state.board[7][7] == '-':
            if state.board[6][7] == my_color:
                my_tiles += 1
            elif state.board[6][7] == opp_color:
                opp_tiles += 1
            if state.board[6][6] == my_color:
                my_tiles += 1
            elif state.board[6][6] == opp_color:
                opp_tiles += 1
            if state.board[7][6] == my_color:
                my_tiles += 1
            elif state.board[7][6] == opp_color:
                opp_tiles += 1
        l = -12.5 * (my_tiles - opp_tiles)

        # Final weighted score
        score = (10 * p) + (801.724 * c) + (382.026 * l) + (74.396 * f) + (10 * d)
        return score
