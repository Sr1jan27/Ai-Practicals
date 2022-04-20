# Ai-Practicals
All Ai experiments at one place in python

**
Toy Problem

TOY 
WATER JUG **

                                                                              -def pour(j1, j2):
                                                                                 m1, m2, f = 2, 5, 3 
                                                                                 print(" %d       %d" % (j1, j2))
                                                                                 if j2 is f:
                                                                                     return
                                                                                 elif j2 is m2:
                                                                                     pour(0, j1)
                                                                                 elif j1 != 0 and j2 is 0:
                                                                                     pour(0, j1)
                                                                                 elif j1 is f:
                                                                                     pour(j1, 0)
                                                                                 elif j1 < m1:
                                                                                     pour(m1, j2)
                                                                                 elif j1 < (m2-j2):
                                                                                     pour(0, (j1+j2))
                                                                                 else:
                                                                                     pour(j1-(m2-j2), (m2-j2)+j2)

                                                                              print("JUG1   JUG2")
                                                                              pour(0, 0)

**
TIC TAC TOE**


                                                   import random


                                                   class TicTacToe:

                                                      def __init__(self):
                                                          self.board = []

                                                      def create_board(self):
                                                          for i in range(3):
                                                              row = []
                                                              for j in range(3):
                                                                  row.append('-')
                                                              self.board.append(row)

                                                      def get_random_first_player(self):
                                                          return random.randint(0, 1)

                                                      def fix_spot(self, row, col, player):
                                                          self.board[row][col] = player

                                                      def is_player_win(self, player):
                                                          win = None

                                                          n = len(self.board)

                                                          # checking rows
                                                          for i in range(n):
                                                              win = True
                                                              for j in range(n):
                                                                  if self.board[i][j] != player:
                                                                      win = False
                                                                      break
                                                              if win:
                                                                  return win

                                                          # checking columns
                                                          for i in range(n):
                                                              win = True
                                                              for j in range(n):
                                                                  if self.board[j][i] != player:
                                                                      win = False
                                                                      break
                                                              if win:
                                                                  return win

                                                          # checking diagonals
                                                          win = True
                                                          for i in range(n):
                                                              if self.board[i][i] != player:
                                                                  win = False
                                                                  break
                                                          if win:
                                                              return win

                                                          win = True
                                                          for i in range(n):
                                                              if self.board[i][n - 1 - i] != player:
                                                                  win = False
                                                                  break
                                                          if win:
                                                              return win
                                                          return False

                                                          for row in self.board:
                                                              for item in row:
                                                                  if item == '-':
                                                                      return False
                                                          return True

                                                      def is_board_filled(self):
                                                          for row in self.board:
                                                              for item in row:
                                                                  if item == '-':
                                                                      return False
                                                          return True

                                                      def swap_player_turn(self, player):
                                                          return 'X' if player == 'O' else 'O'

                                                      def show_board(self):
                                                          for row in self.board:
                                                              for item in row:
                                                                  print(item, end=" ")
                                                              print()

                                                      def start(self):
                                                          self.create_board()

                                                          player = 'X' if self.get_random_first_player() == 1 else 'O'
                                                          while True:
                                                              print(f"Player {player} turn")

                                                              self.show_board()

                                                              # taking user input
                                                              row, col = list(
                                                                  map(int, input("Enter row and column numbers to fix spot: ").split()))
                                                              print()

                                                              # fixing the spot
                                                              self.fix_spot(row - 1, col - 1, player)

                                                              # checking whether current player is won or not
                                                              if self.is_player_win(player):
                                                                  print(f"Player {player} wins the game!")
                                                                  break

                                                              # checking whether the game is draw or not
                                                              if self.is_board_filled():
                                                                  print("Match Draw!")
                                                                  break

                                                              # swapping the turn
                                                              player = self.swap_player_turn(player)

                                                          # showing the final view of board
                                                          print()
                                                          self.show_board()


                                                   # starting the game
                                                   tic_tac_toe = TicTacToe()
                                                   tic_tac_toe.start()


**SUDOKU
**
                                                                        # N is the size of the 2D matrix N*N
                                                                        N = 9


                                                                        # A utility function to print grid
                                                                        def printing(arr):
                                                                           for i in range(N):
                                                                               for j in range(N):
                                                                                   print(arr[i][j], end=" ")
                                                                               print()


                                                                        # Checks whether it will be
                                                                        # legal to assign num to the
                                                                        # given row, col
                                                                        def isSafe(grid, row, col, num):
                                                                           # Check if we find the same num
                                                                           # in the similar row , we
                                                                           # return false
                                                                           for x in range(9):
                                                                               if grid[row][x] == num:
                                                                                   return False

                                                                           # Check if we find the same num in
                                                                           # the similar column , we
                                                                           # return false
                                                                           for x in range(9):
                                                                               if grid[x][col] == num:
                                                                                   return False

                                                                           # Check if we find the same num in
                                                                           # the particular 3*3 matrix,
                                                                           # we return false
                                                                           startRow = row - row % 3
                                                                           startCol = col - col % 3
                                                                           for i in range(3):
                                                                               for j in range(3):
                                                                                   if grid[i + startRow][j + startCol] == num:
                                                                                       return False
                                                                           return True


                                                                        # Takes a partially filled-in grid and attempts
                                                                        # to assign values to all unassigned locations in
                                                                        # such a way to meet the requirements for
                                                                        # Sudoku solution (non-duplication across rows,
                                                                        # columns, and boxes) */
                                                                        def solveSudoku(grid, row, col):
                                                                           # Check if we have reached the 8th
                                                                           # row and 9th column (0
                                                                           # indexed matrix) , we are
                                                                           # returning true to avoid
                                                                           # further backtracking
                                                                           if (row == N - 1 and col == N):
                                                                               return True

                                                                           # Check if column value becomes 9 ,
                                                                           # we move to next row and
                                                                           # column start from 0
                                                                           if col == N:
                                                                               row += 1
                                                                               col = 0

                                                                           # Check if the current position of
                                                                           # the grid already contains
                                                                           # value >0, we iterate for next column
                                                                           if grid[row][col] > 0:
                                                                               return solveSudoku(grid, row, col + 1)
                                                                           for num in range(1, N + 1, 1):

                                                                               # Check if it is safe to place
                                                                               # the num (1-9) in the
                                                                               # given row ,col ->we
                                                                               # move to next column
                                                                               if isSafe(grid, row, col, num):

                                                                                   # Assigning the num in
                                                                                   # the current (row,col)
                                                                                   # position of the grid
                                                                                   # and assuming our assigned
                                                                                   # num in the position
                                                                                   # is correct
                                                                                   grid[row][col] = num

                                                                                   # Checking for next possibility with next
                                                                                   # column
                                                                                   if solveSudoku(grid, row, col + 1):
                                                                                       return True

                                                                               # Removing the assigned num ,
                                                                               # since our assumption
                                                                               # was wrong , and we go for
                                                                               # next assumption with
                                                                               # diff num value
                                                                               grid[row][col] = 0
                                                                           return False


                                                                        # Driver Code

                                                                        # 0 means unassigned cells
                                                                        grid = [[3, 0, 6, 5, 0, 8, 4, 0, 0],
                                                                               [5, 2, 0, 0, 0, 0, 0, 0, 0],
                                                                               [0, 8, 7, 0, 0, 0, 0, 3, 1],
                                                                               [0, 0, 3, 0, 1, 0, 0, 8, 0],
                                                                               [9, 0, 0, 8, 6, 3, 0, 0, 5],
                                                                               [0, 5, 0, 0, 9, 0, 6, 0, 0],
                                                                               [1, 3, 0, 0, 0, 0, 2, 5, 0],
                                                                               [0, 0, 0, 0, 0, 0, 0, 7, 4],
                                                                               [0, 0, 5, 2, 0, 6, 3, 0, 0]]

                                                                        if (solveSudoku(grid, 0, 0)):
                                                                           printing(grid)
                                                                        else:
                                                                           print("no solution exists ")
**

REAL WORLD PROBLEM


REAL WORLD PROBLEM 
EXAMPLE :} Vacuum **


                                                                                    import random
                                                                                    class Environment(object):
                                                                                       def init (self):
                                                                                           self.locationCondition = {'A': '0', 'B': '0'}
                                                                                           self.locationCondition['A'] = random.randint(0, 1)
                                                                                           self.locationCondition['B'] = random.randint(0, 1)
                                                                                    class SimpleReflexVacuumAgent(Environment):
                                                                                       def init (self, Environment):
                                                                                           print (Environment.locationCondition)
                                                                                           Score = 0
                                                                                           vacuumLocation = random.randint(0, 1)
                                                                                           if vacuumLocation == 0:
                                                                                               print ("Vacuum is randomly placed at Location A")
                                                                                               if Environment.locationCondition['A'] == 1:
                                                                                                   print ("Location A is Dirty. ")
                                                                                                   Environment.locationCondition['A'] = 0;
                                                                                                   Score += 1
                                                                                                   print ("Location A has been Cleaned. :D")
                                                                                                   if Environment.locationCondition['B'] == 1:
                                                                                                       print ("Location B is Dirty.")
                                                                                                       print ("Moving to Location B...")
                                                                                                       Score -= 1
                                                                                                       Environment.locationCondition['B'] = 0;
                                                                                                       Score += 1
                                                                                                       print ("Location B has been Cleaned :D.")
                                                                                               else:
                                                                                                   if Environment.locationCondition['B'] == 1:
                                                                                                       print ("Location B is Dirty.")
                                                                                                       Score -= 1
                                                                                                       print ("Moving to Location B...")
                                                                                                       Environment.locationCondition['B'] = 0;
                                                                                                       Score += 1
                                                                                                       print ("Location B has been Cleaned. :D")
                                                                                           elif vacuumLocation == 1:
                                                                                               print ("Vacuum is randomly placed at Location B. ")
                                                                                               if Environment.locationCondition['B'] == 1:
                                                                                                   print ("Location B is Dirty")
                                                                                                   Environment.locationCondition['B'] = 0;
                                                                                                   Score += 1
                                                                                                   print ("Location B has been Cleaned")
                                                                                                   if Environment.locationCondition['A'] == 1:
                                                                                                       print ("Location A is Dirty")
                                                                                                       Score -= 1
                                                                                                       print ("Moving to Location A")
                                                                                                       Environment.locationCondition['A'] = 0;
                                                                                                       Score += 1
                                                                                                       print ("Location A has been Cleaned")
                                                                                           else:
                                                                                               if Environment.locationCondition['A'] == 1:
                                                                                                   print ("Location A is Dirty")
                                                                                                   print ("Moving to Location A")
                                                                                                   Score -= 1
                                                                                                   Environment.locationCondition['A'] = 0;
                                                                                                   Score += 1
                                                                                                   print ("Location A has been Cleaned")
                                                                                           print (Environment.locationCondition)
                                                                                           print ("Performance Measurement: " + str(Score))
                                                                                    theEnvironment = Environment()
                                                                                    theEnvironment.init()
                                                                                    theVacuum = SimpleReflexVacuumAgent()
                                                                                    theVacuum.init(theEnvironment)




**
CSP  N QUEENS



CSP 
Example N QUEEN :-
**

                                                                               class NQueens:
                                                                                 def __init__(self,n ):
                                                                                     self.n = n
                                                                                     self.chess_table = [[0 for i in range(n)]for j in range(n)]

                                                                                 def print_Queens(self):
                                                                                     for i in range (self.n):
                                                                                         for j in range (self.n):
                                                                                             if self.chess_table[i][j] == 1:
                                                                                                 print(" Q ", end =' ')
                                                                                             else:
                                                                                                 print(" - ",end = ' ')
                                                                                         print("\n")

                                                                                 def is_place_safe(self,row_index,col_index ):
                                                                                     for i in range(self.n):
                                                                                         if self.chess_table[row_index][i] == 1:
                                                                                             return False

                                                                                     j = col_index
                                                                                     for i in range(row_index,-1,-1):
                                                                                         if i<0:
                                                                                             break
                                                                                         if self.chess_table[i][j] == 1:
                                                                                             return False
                                                                                         j = j - 1

                                                                                     j = col_index
                                                                                     for i in range(row_index,self.n):
                                                                                         if i<0:
                                                                                             break
                                                                                         if self.chess_table[i][j] == 1:
                                                                                             return False
                                                                                         j=j-1

                                                                                     return True

                                                                                 def solve(self,col_index ):

                                                                                     if col_index == self.n:
                                                                                         return  True

                                                                                     for row_index in range(self.n):
                                                                                         if self.is_place_safe(row_index,col_index):
                                                                                             self.chess_table[row_index][col_index] = 1
                                                                                             if self.solve(col_index+1):
                                                                                                 return True
                                                                                             self.chess_table[row_index][col_index] = 0

                                                                                     return False

                                                                                 def solveN_Queens(self):
                                                                                     if self.solve(0):
                                                                                         self.print_Queens()
                                                                                     else:
                                                                                         print("no solution to the problem")







                                                                              queens = NQueens(4)
                                                                              queens.solveN_Queens()







**BFS DFS




BFS DFS
Code
**


                                                                              graph = {
                                                                              'S': ['A', 'B'],
                                                                              'A': ['C', 'D'],
                                                                              'B': ['G','H'],
                                                                              'C': ['E','F'], 'D': [],
                                                                              'G': ['I'],
                                                                              'H': [],
                                                                              'E': ['K'],
                                                                              'F': [],
                                                                              'I': [],
                                                                              'K': []
                                                                              }
                                                                              visitd =[]
                                                                              queue=[]
                                                                              visited1 =[]
                                                                              def bfs(visitd,graph,node):
                                                                                 visitd.append(node)
                                                                                 queue.append(node)

                                                                                 while queue:
                                                                                     P=queue.pop(0)
                                                                                     print(P,end=" ")

                                                                                     for neighbour in graph[P]:
                                                                                         if neighbour not in visitd:
                                                                                             visitd.append(neighbour)
                                                                                             queue.append(neighbour)
                                                                              visited = set()
                                                                              def dfs(visited, graph, node):
                                                                                 if node not in visited:
                                                                                     print (node)
                                                                                     visited.add(node)
                                                                                     for neighbour in graph[node]:
                                                                                         dfs(visited, graph, neighbour)
                                                                              print("Breadth first search")
                                                                              bfs(visitd,graph,'S')
                                                                              print()
                                                                              print("Following is the Depth-First Search")
                                                                              dfs(visited, graph, 'S')







**
MIN MAX





MIN MAX 
Code**


                                                                                    import math
                                                                                    def minimax(curDepth, nodeIndex,
                                                                                               maxTurn, scores,
                                                                                               targetDepth):
                                                                                       if (curDepth == targetDepth):
                                                                                           return scores[nodeIndex]

                                                                                       if (maxTurn):
                                                                                           return max(minimax(curDepth + 1, nodeIndex * 2,
                                                                                                              False, scores, targetDepth),
                                                                                                      minimax(curDepth + 1, nodeIndex * 2 + 1,
                                                                                                              False, scores, targetDepth))

                                                                                       else:
                                                                                           return min(minimax(curDepth + 1, nodeIndex * 2,
                                                                                                              True, scores, targetDepth),
                                                                                                      minimax(curDepth + 1, nodeIndex * 2 + 1,
                                                                                                              True, scores, targetDepth))


                                                                                    scores = [3, 5, 2, 9, 12, 5, 23, 23]

                                                                                    treeDepth = math.log(len(scores), 2)

                                                                                    print(scores)
                                                                                    print("The optimal value is : ", end="")
                                                                                    print(minimax(0, 0, True, scores, treeDepth))







**
BST A*

BST **


                                                                                    from queue import PriorityQueue

                                                                                    # Filling adjacency matrix with empty arrays
                                                                                    vertices = 14
                                                                                    graph = [[] for i in range(vertices)]


                                                                                    # Function for adding edges to graph
                                                                                    def add_edge(x, y, cost):
                                                                                       graph[x].append((y, cost))
                                                                                       graph[y].append((x, cost))


                                                                                    # Function For Implementing Best First Search
                                                                                    # Gives output path having the lowest cost
                                                                                    def best_first_search(source, target, vertices):
                                                                                       visited = [0] * vertices
                                                                                       pq = PriorityQueue()
                                                                                       pq.put((0, source))
                                                                                       print("Path: ")
                                                                                       while not pq.empty():
                                                                                           u = pq.get()[1]
                                                                                           # Displaying the path having the lowest cost
                                                                                           print(u, end=" ")
                                                                                           if u == target:
                                                                                               break

                                                                                           for v, c in graph[u]:
                                                                                               if not visited[v]:
                                                                                                   visited[v] = True
                                                                                                   pq.put((c, v))
                                                                                       print()


                                                                                    if __name__ == '__main__':
                                                                                       # The nodes shown in above example(by alphabets) are
                                                                                       # implemented using integers add_edge(x,y,cost);
                                                                                       add_edge(0, 1, 1)
                                                                                       add_edge(0, 2, 8)
                                                                                       add_edge(1, 2, 12)
                                                                                       add_edge(1, 4, 13)
                                                                                       add_edge(2, 3, 6)
                                                                                       add_edge(4, 3, 3)

                                                                                       source = 0
                                                                                       target = 2
                                                                                       best_first_search(source, target, vertices)

A****************
                                                            from collections import deque

                                                            class Graph:
                                                               # example of adjacency list (or rather map)
                                                               # adjacency_list = {
                                                               # 'A': [('B', 1), ('C', 3), ('D', 7)],
                                                               # 'B': [('D', 5)],
                                                               # 'C': [('D', 12)]
                                                               # }

                                                               def __init__(self, adjacency_list):
                                                                   self.adjacency_list = adjacency_list

                                                               def get_neighbors(self, v):
                                                                   return self.adjacency_list[v]

                                                               # heuristic function with equal values for all nodes
                                                               def h(self, n):
                                                                   H = {
                                                                       'A': 1,
                                                                       'B': 1,
                                                                       'C': 1,
                                                                       'D': 1
                                                                   }

                                                                   return H[n]

                                                               def a_star_algorithm(self, start_node, stop_node):
                                                                   # open_list is a list of nodes which have been visited, but who's neighbors
                                                                   # haven't all been inspected, starts off with the start node
                                                                   # closed_list is a list of nodes which have been visited
                                                                   # and who's neighbors have been inspected
                                                                   open_list = set([start_node])
                                                                   closed_list = set([])

                                                                   # g contains current distances from start_node to all other nodes
                                                                   # the default value (if it's not found in the map) is +infinity
                                                                   g = {}

                                                                   g[start_node] = 0

                                                                   # parents contains an adjacency map of all nodes
                                                                   parents = {}
                                                                   parents[start_node] = start_node

                                                                   while len(open_list) > 0:
                                                                       n = None

                                                                       # find a node with the lowest value of f() - evaluation function
                                                                       for v in open_list:
                                                                           if n == None or g[v] + self.h(v) < g[n] + self.h(n):
                                                                               n = v;

                                                                       if n == None:
                                                                           print('Path does not exist!')
                                                                           return None

                                                                       # if the current node is the stop_node
                                                                       # then we begin reconstructin the path from it to the start_node
                                                                       if n == stop_node:
                                                                           reconst_path = []

                                                                           while parents[n] != n:
                                                                               reconst_path.append(n)
                                                                               n = parents[n]

                                                                           reconst_path.append(start_node)

                                                                           reconst_path.reverse()

                                                                           print('Path found: {}'.format(reconst_path))
                                                                           return reconst_path

                                                                       # for all neighbors of the current node do
                                                                       for (m, weight) in self.get_neighbors(n):
                                                                           # if the current node isn't in both open_list and closed_list
                                                                           # add it to open_list and note n as it's parent
                                                                           if m not in open_list and m not in closed_list:
                                                                               open_list.add(m)
                                                                               parents[m] = n
                                                                               g[m] = g[n] + weight

                                                                           # otherwise, check if it's quicker to first visit n, then m
                                                                           # and if it is, update parent data and g data
                                                                           # and if the node was in the closed_list, move it to open_list
                                                                           else:
                                                                               if g[m] > g[n] + weight:
                                                                                   g[m] = g[n] + weight
                                                                                   parents[m] = n

                                                                                   if m in closed_list:
                                                                                       closed_list.remove(m)
                                                                                       open_list.add(m)

                                                                       # remove n from the open_list, and add it to closed_list
                                                                       # because all of his neighbors were inspected
                                                                       open_list.remove(n)
                                                                       closed_list.add(n)

                                                                   print('Path does not exist!')
                                                                   return None
                                                            adjacency_list = {
                                                               'A': [('B', 1), ('C', 3), ('D', 7)],
                                                               'B': [('D', 5)],
                                                               'C': [('D', 12)]
                                                            }
                                                            graph1 = Graph(adjacency_list)
                                                            graph1.a_star_algorithm('A', 'D')









UNCERTAIN PROBLEMS


Uncertain methods 

Problem 1


                                                            import numpy as np import collections
                                                            npArray= np.array([60, 70, 70, 70, 80,90,60])
                                                            c=collections.Counter(npArray) # Generate a dictionary {"value":"nbOfOccurrences"}
                                                            arraySize=npArray.size
                                                            nbOfOccurrences=c[60] #assuming you want the proba to get 10
                                                            proba=(nbOfOccurrences/arraySize)*100
                                                            print(proba) #print 60.0

Output 28.57









