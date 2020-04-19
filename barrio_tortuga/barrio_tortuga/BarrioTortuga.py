"""
Definition of Barrio Tortuga

### Madrid central district:

- Barrio tortuga is modelled as an idealisation of Madrid central district, a neighborhood
(barrio) characterised by:

- Popultion: 131,928 people
- Area: 5.23 km2.
- Number of home: 63,622
- Thus number of people per home ~2

**Barrio Tortuga**

- Toroidal grid of 100 x 100 m2
- Made of square buildings of 30 x 30 m2.
- Between buildings run avenues of 20 m.
- Thus the grid has 4 buildings, 2 avenues running longitudinal and 2 avenues running
horizontal.
- The inhabitable are is 0.6 x 0.6 = 0.36 of the total area
- We assume that each building has 30 apartments. Each apartment has 30 m2.
- We can assume that each building is 10 stories high.
- Thus the number of homes in the patch of the barrio is 30 x 4 = 120 homes.
- The number of turtles in barrio tortuga is 252 (same density that in Madrid center)
   and thus the number of turtles per home is also 2.
"""

from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa import Agent
from mesa.time import RandomActivation
import numpy as np

from enum import Enum
class PrtLvl(Enum):
    Mute     = 1
    Concise  = 2
    Detailed = 3
    Verbose  = 4

### AGENTS

class Patch(Agent):
    """
    A urban patch (land) which does not move.
    kind = 1 means the patch belongs to a building
    kind = 2 means the patch belongs to an avenue
    """
    def __init__(self, unique_id, pos, model, kind, moore=False):
        super().__init__(unique_id, model)
        self.kind = kind
        self.pos  = pos
        self.moore  = moore

    def step(self):
        pass


class Turtle(Agent):
    '''
    A turtle able to move in streets
    '''
    def __init__(self, unique_id, pos, model, moore=True):
        super().__init__(unique_id, model)
        self.pos = pos
        self.moore = moore

    def check_if_in_street(self, pos):
        x,y = pos
        if self.model.map_bt[x,y] == 2:
            return True
        else:
            return False

    def move(self):
        # Get neighborhood
        neighbors = [i for i in self.model.grid.get_neighborhood(self.pos, self.moore,
                False)]

        #print(f'Turtle in position {self.pos}: neighbors ={neighbors}')
        allowed_neighbors = []
        for pos in neighbors:
            if self.check_if_in_street(pos):
                allowed_neighbors.append(pos)

        self.random.shuffle(allowed_neighbors)
        self.model.grid.move_agent(self, allowed_neighbors[0])


    def step(self):
        self.move()


# MODEL

prtl =PrtLvl.Concise

def print_level(prtl, PRT):
    if prtl.value >= PRT.value:
        return True
    else:
        return False

def number_of_encounters(model):
    nc = 0
    for y in range(model.grid.height):
        for x in range(model.grid.width):
            this_cell = model.grid.get_cell_list_contents((x,y))
            n_turtles = len([obj for obj in this_cell if isinstance(obj, Turtle)])
            if n_turtles > 0:
                if print_level(prtl, PrtLvl.Verbose):
                    print(f'number of turtles found in {x,y} -->{ n_turtles}')
            if n_turtles > 1:
                nc += 1

    if print_level(prtl, PrtLvl.Detailed):
        print(f'total number of encounters this step ={nc}')
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

    return nc

class BarrioTortuga(Model):
    '''
    A neighborhood where people goes out of their homes, walk around at random and meet other people.
    '''

    def __init__(self, map_file="barrio-tortuga-map-dense.txt", turtles=250, nd=1, prtl=PrtLvl.Detailed):
        '''
        Create a new Barrio Tortuga.

        The barrio is created from a map. It is a toroidal object thus moore is always True
        The barrio is fille with turtles that can exist the buildings through a set of doors.
        There are many doors in the building. This is meant to represent the temporal spread in
        the turtles coming in and out of the buildings. In a real building persons go out by the same
        door at different times. In Barrio Tortugas, turtles go out of the buildings through a set
        of doors. Each turtle chooses a door to go out at random. This is equivalent to introduce
        randomness on the exit time.

        Args:
            The  name file with the barrio map
            number of turtles in the barrio
            nd, a parameter that decides the number of doors (largest for nd=1)
        '''

        # read the map
        self.map_bt                 = np.genfromtxt(map_file)

        if print_level(prtl, PrtLvl.Concise):
            print(f'loaded barrio tortuga map with dimensions ->{ self.map_bt.shape}')

        self.height, self.width     = self.map_bt.shape
        self.grid                   = MultiGrid(self.height, self.width, torus=True)
        self.moore                  = True
        self.turtles                = turtles
        self.schedule               = RandomActivation(self)
        self.datacollector          = DataCollector(
        model_reporters             = {"NumberOfEncounters": number_of_encounters}
        )

        # create the patches representing houses and avenues
        id = 0
        for _, x, y in self.grid.coord_iter():
            patch_kind = self.map_bt[x, y]               # patch kind labels buildings or streets
            patch = Patch(id, (x, y), self, patch_kind)
            self.grid.place_agent(patch, (x, y))         # agents are placed in the grid but not in the
                                                         # in the schedule

        # Create turtles distributed randomly in the doors
        doors = self.get_doors(nd)
        if print_level(prtl, PrtLvl.Detailed):
            print(f'doors = {doors}')

        n_doors = len(doors)
        if print_level(prtl, PrtLvl.Concise):
            print(f'number of doors = {n_doors}')

        for i in range(self.turtles):
            n = self.random.randrange(n_doors)  # choose the door
            d = doors[n]
            x=  d[0]
            y = d[1]                    # position of the door
            if print_level(prtl, PrtLvl.Detailed):
                print(f'starting turtle {i} at door number {n}, x,y ={x,y}')

            a = Turtle(i, (x, y), self, True)  # create Turtle

            self.schedule.add(a)               # add to scheduler
            self.grid.place_agent(a, (x, y))   # place Turtle in the grid


        self.running = True

        # activate data collector
        self.datacollector.collect(self)

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)

    def get_doors(self, n=1):
        D = []
        for j in range (0,7):
            pdl = np.array((10, 5 + j * 15))
            pdr = np.array((14, 5 + j *15))
            print(f' raw = {j}, pdl ={pdl}, pdr = {pdr}')
            R0DL = [pdl + i * np.array((15,0)) for i in range(0,6)]
            R0DR = [pdr + i * np.array((15,0)) for i in range(0,6)]
            print('doors')
            print(R0DL)
            print(R0DR)
            for r in R0DL:
                D.append(r)
            for r in R0DR:
                D.append(r)
        return D#
#     def get_doors(self, n=1):
#     D = []
#     for x in range(0,31,n):
#         D.append((x,30))
#         D.append((30,x))
#         D.append((49,x))
#         D.append((x,49))
#         D.append((80,x))
#         D.append((x,80))
#
#     for x in range(51,81,n):
#         D.append((x,30))
#         D.append((x,49))
#         D.append((x,80))
#
#     for x in range(49,81,n):
#         D.append((30,x))
#         D.append((49,x))
#
#     # for x in range(81,99,n):
#     #     D.append((x,49))
#     return D


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
