from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa import Agent
from mesa.time import RandomActivation

def number_of_turtles_in_cell(model):
    for y in range(model.grid.height):
        for x in range(model.grid.width):
            nc = 0
            c = model.grid.get_cell_list_contents((x,y))
            nc += len(c)

    #print(f'step {model.step}, nc = {nc}')
    return nc

def number_of_turtles_in_neighborhood(model):
    for y in range(model.grid.height):
        for x in range(model.grid.width):
            c = model.grid.get_cell_list_contents((x,y))
            if len(c) > 0:
                n_xy = model.grid.get_neighborhood((x,y), model.moore, False) #coordinates of neighbors
                nc = 0
                for xy in n_xy:
                    c = model.grid.get_cell_list_contents((xy[0], xy[1]))
                    nc += len(c)

    #print(f'step {model.step}, nc = {nc}')
    return nc


class BarrioTortuga(Model):
    '''
    A neighborhood where people goes out of their homes, walk around at random and meet other people.
    '''

    def __init__(self, height=50, width=50, turtles=10, moore=True):
        '''
        Create a new Barrio Tortuga

        Args:
            height, width: World size.
            agent_count: How many agents to create.
        '''


        self.height       = height
        self.width        = width
        self.grid         = MultiGrid(self.height, self.width, torus=True)
        self.agent_count  = turtles
        self.schedule     = RandomActivation(self)
        self.moore        = moore
        self.datacollector = DataCollector(
        model_reporters={"NumberOfTurtlesPerCell": number_of_turtles_in_cell,
                         "NumberOfTurtlesNeighborhood": number_of_turtles_in_neighborhood}
        )



        # Create agents
        for i in range(self.agent_count):
            x, y = self.random_xy_grid()         # random x, y positions in the grid
            a = RandomTurtle(i, (x, y), self, True)  # create Turtle

            self.schedule.add(a)               # add to scheduler
            self.grid.place_agent(a, (x, y))   # place Turtle in the grid

        self.running = True
        # activate data collector
        self.datacollector.collect(self)

    def random_xy_grid(self):
        x = self.random.randrange(self.width)       # random x, y positions in the grid
        y = self.random.randrange(self.height)
        return x, y

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)


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
