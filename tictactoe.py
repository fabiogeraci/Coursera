import random

class Player:

    def __init__(self, marker='X', is_human=True):
        self._marker = marker
        self._is_human = is_human

    @property
    def marker(self):
        return self._marker

    @property
    def is_human(self):
        return self._is_human

    def get_player_move(self):
        if self._is_human:
            return self.get_human_move()
        else:
            return self.get_computer_move()

    def get_human_move(self):
        move = input('Player Move (X): ')
        return move

    def get_computer_move(self):
        row = random.choice([1, 2, 3])
        col = random.choice(['A', 'B', 'C'])
        move = str(row)+col

        print('Computer Move (O): ', move)
        return move


class Board:

    EMPTY = 0
    COLUMNS = {'A': 0, 'B': 1, 'C': 2}
    ROWS = (1, 2, 3)

    def __init__(self, game_board=None):
        if game_board:
            self.game_board = game_board
        else:
            self.game_board = [[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]]

    def print_board(self):
        print('    A   B   C')
        for i, row in enumerate(self.game_board, 1):
            print(str(i), end=' | ')
            for col in row:
                if col != Board.EMPTY:
                    print(col, end=' | ')
                else:
                    print('  | ', end='')
            print('\n---------------')

    def submit_move(self, move, player):
        if not self.is_move_valid(move):
            print('Invalid Imput: Please enter the row and column of '
                  'your move ')
            return
        else:
            row_index = int(move[0]) -1
            col_index = Board.COLUMNS[move[1]]

            value = self.game_board[row_index][col_index]

            if value == Board.EMPTY:
                self.game_board[row_index][col_index] = player.marker
            else:
                print('This space is already taken.')

    def is_move_valid(self, move):
        return ((len(move) == 2)
            and (int(move[0]) in Board.ROWS)
            and (move[1] in Board.COLUMNS))


    def is_winner(self, player, row, col):
        if self.check_row(row, player):
            return True
        elif self.check_col(col, player):
            return True
        elif self.check_diagonal(player):
            return True
        elif self.check_antidiagonal(player):
            return True
        else:
            return False

    def check_row(self, row, player):
        row_index = int(row) - 1
        board_row = self.game_board[row_index]

        if board_row.count(player.marker) == 3:
            return True
        else:
            return False

    def check_col(self, col, player):
        col_index = Board.COLUMNS[col]
        total_markers = 0

        for i in range(3):
            if self.game_board[i][col_index] == player.marker:
                total_markers += 1

        if total_markers == 3:
            return True
        else:
            return False

    def check_diagonal(self, player):
        total_markers = 0

        for i in range(3):
            if self.game_board[i][i] == player.marker:
                total_markers +=1

        if total_markers == 3:
            return True
        else:
            return False

    def check_antidiagonal(self, player):
        total_markers = 0

        for i in range(3):
            if self.game_board[i][2-i] == player.marker:
                total_markers += 1

        if total_markers == 3:
            return True
        else:
            return False

    def check_tie(self):
        total_empty = 0

        for row in self.game_board:
            total_empty += row.count(Board.EMPTY)

        if total_empty == 0:
            return True
        else:
            return False


print('****************')
print('  Tic-Tac-Toe!')
print('****************')

board = Board()
player = Player()
computer = Player('O', False)

board.print_board()

while True:
    # Ask Human User Move
    move = player.get_player_move()
    # Submit Move
    board.submit_move(move, player)
    # Print Board
    board.print_board()

    if board.is_move_valid(move) and board.is_winner(player, move[0], move[1]):
        print('You Win')
        break

    # Ask Computer Move
    comp_move = computer.get_player_move()
    # Submit Move
    board.submit_move(comp_move, computer)
    # Print Board
    board.print_board()

    if board.is_winner(computer, comp_move[0], comp_move[1]):
        print('Computer Win')
        break

    if board.check_tie():
        print('Tie Game')
        break
    else:
        print('Carry on')


