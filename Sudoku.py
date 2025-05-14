import numpy
import copy

class SudokuSolver:
	def __init__(self, grid):
		self.grid = grid
		self.solutions = []

	def is_number_valid(self, row, col, number):

		in_row = number in self.grid[row, : ]
		in_col = number in self.grid[:, col]
		
		square_row = (row//3) * 3
		square_col = (col//3) * 3 
		in_square = number in self.grid[square_row: square_row+3, square_col : square_col+3]

		number_is_valid = not (in_row or in_col or in_square)
		# number_is_valid = not in_row and not in_col and not in_square
		
		return number_is_valid



	def solve(self):
		for row in range(0, 9):
			for col in range(0, 9):
				if self.grid[row][col] == 0 :
					for possible_value in range (1,10):
						if self.is_number_valid(row, col, possible_value):
							self.grid[row][col] = possible_value
							self.solve()
							self.grid[row][col] = 0
					return None
		else:
			print(self.grid)


if __name__ == '__main__':
	grid = numpy.array([[0, 0, 6, 5, 3, 0, 0, 1, 0],
						[0, 0, 0, 0, 2, 0, 5, 0, 4],
						[2, 0, 0, 0, 0, 0, 0, 0, 0],
						[0, 0, 0, 0, 8, 0, 3, 4, 2],
						[0, 0, 0, 9, 0, 3, 0, 0, 0],
						[6, 3, 1, 0, 4, 0, 0, 0, 0],
						[0, 0, 0, 0, 0, 0, 0, 0, 7],
						[8, 0, 4, 0, 6, 0, 0, 0, 0],
						[0, 6, 0, 0, 7, 1, 8, 0, 0]]) 

	sudoku_object = SudokuSolver(grid)
	sudoku_object.solve()
	