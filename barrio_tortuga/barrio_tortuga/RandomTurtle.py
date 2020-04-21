class WalkingAgent(Agent):
    '''
    Class implementing a turtle that can move at random

    Not indended to be used on its own, but to inherit its methods to multiple
    other agents.

    '''

    def __init__(self, unique_id, pos, model, moore=True):
        '''
        grid: The MultiGrid object in which the agent lives.
        x: The agent's current x coordinate
        y: The agent's current y coordinate
        moore: If True, may move in all 8 directions.
                Otherwise, only up, down, left, right.
        '''
        super().__init__(unique_id, model)
        self.pos = pos
        self.moore = moore

    def random_move(self):
        '''
        Step one cell in any allowable direction.
        We use grid’s built-in get_neighborhood method, which returns all the neighbors of a given cell.
        This method can get two types of cell neighborhoods: Moore (including diagonals),
        and Von Neumann (only up/down/left/right).
        It also needs an argument as to whether to include the center cell itself as one of the neighbors.
        '''
        # Pick the next cell from the adjacent cells.
        next_moves = self.model.grid.get_neighborhood(self.pos, self.moore, True)
        next_move = self.random.choice(next_moves)
        # Now move:
        self.model.grid.move_agent(self, next_move)


class RandomTurtle(WalkingAgent):
    '''
    Agent which only walks around.
    '''

    def step(self):
        self.random_move()
