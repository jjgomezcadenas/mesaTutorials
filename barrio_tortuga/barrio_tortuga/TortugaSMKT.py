"""
Definition of Tortuga Supermarket

#Tortuga SMKT is modelled as an idealisation of a supermarket characterised by:

- Max people in SMKT: 120 (can be regulated)
- Area: 1600 m2.

**Barrio Tortuga SMKT**

- Non Toroidal grid of 40 x 4s0 m2
- grey patches represent walls: turtles cannot walk there.
- blue patches represent courridors for turtles to walk.
- red patches are hot spots (bounlagerie, fish, meat...).
A turtle which has a hot spot patch as a neighbour has a probability ph to stay put in a given tick.
- green patches are payment spots. A turtle reaching there stays for a while
 (pp, larger than ph) then goes around through the blue patches to exit.
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


def print_level(prtl, PRT):
    if prtl.value >= PRT.value:
        return True
    else:
        return False


prtl =PrtLvl.Concise


def is_courridor(map_bt, x, y):
    if map_bt[x,y] == 2:
        return True
    else:
        return False

def is_wall(map_bt, x, y):
    if map_bt[x,y] == 1:
        return True
    else:
        return False

def is_hot_spot(map_bt, x, y):
    if map_bt[x,y] == 3:
        return True
    else:
        return False

def is_payment(map_bt, x, y):
    if map_bt[x,y] == 4:
        return True
    else:
        return False
### AGENTS

class Patch(Agent):
    """
    A supermarket patch.
    kind = 1 means the patch belongs to a wall
    kind = 2 means the patch belongs to courridor
    kind = 3 means the patch belongs to a hot spot
    kind = 4 means the patch belongs to payment booth
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


    def avoid_turtle(self):
        atry =  np.random.random_sample() # check awareness
        if atry < self.model.avoid_awareness:
            return True
        else:
            return False


    def socialize_turtle(self):
        atry =  np.random.random_sample() # check awareness
        if atry < self.model.social_affinity:
            return True
        else:
            return False


    def filled_with_turtles(self, cell):
        turtles = [obj for obj in cell if isinstance(obj, Turtle)]
        if len(turtles) > 0:
            return True
        else:
            return False

    def move(self):
        # Get neighborhood
        neighbors = [i for i in self.model.grid.get_neighborhood(self.pos, self.moore,
                False)]

        allowed_neighbors = []
        for pos in neighbors:
            if self.check_if_in_street(pos):
                allowed_neighbors.append(pos)

        selected_neighbors = []

        if self.model.social_affinity == 0:  # just move at random
            selected_neighbors = allowed_neighbors
        elif self.model.social_affinity > 0:  # Try to move into an occupied cell if you can
            for pos in allowed_neighbors:
                n_cell = self.model.grid.get_cell_list_contents([pos])
                if self.filled_with_turtles(n_cell) == True:
                    if self.socialize_turtle() == True:  # pass test
                        selected_neighbors.append(pos)   # select spot as candidate

            if len(selected_neighbors) == 0:  # no spot with another turtle
                selected_neighbors = allowed_neighbors # thus move at random to available

        elif self.model.social_affinity < 0:  # Try to avoid occupied cells if you can
            for pos in allowed_neighbors:
                n_cell = self.model.grid.get_cell_list_contents([pos])
                if self.filled_with_turtles(n_cell) == True:
                    if self.avoid_turtle() == False:  # failed avoiding turtle
                        selected_neighbors.append(pos)
                else:
                    selected_neighbors.append(pos) # this one is void, OK

            if len(selected_neighbors) == 0:  # no spot are voids
                selected_neighbors = allowed_neighbors # thus move at random to available

        self.random.shuffle(selected_neighbors)
        self.model.grid.move_agent(self, selected_neighbors[0])


    def step(self):
        self.move()


# MODEL
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
    A neighborhood where turtles goes out of their homes, walk around at random
    and meet other turtles.

    '''

    def __init__(self,
                 map_file="barrio-tortuga-map-dense.txt",
                 turtles=250,
                 social_affinity = 0.,
                 nd=2,
                 prtl=PrtLvl.Detailed):
        '''
        Create a new Barrio Tortuga.

        The barrio is created from a map. It is a toroidal object thus moore is always True
        The barrio is fille with turtles that can exist the buildings through a set of doors.
        There are many doors in the building. This is meant to represent the temporal spread in
        the turtles coming in and out of the buildings.

        In a real building persons go out by the same
        door at different times. In Barrio Tortugas, turtles go out of the buildings through a set
        of doors. Each turtle chooses a door to go out at random. This is equivalent to introduce
        randomness on the exit time.

        Args:
            The  name file with the barrio map
            number of turtles in the barrio
            social_affinity a parameter that sets the social affinity of turtles. It takes
            values between -1 and 1. For positive values (0, 1), turtles seek contact with
            other turtles. For negative values (-1, 0), turtles avoid contact with others.
            A value of social_affinity of 1 means that a turtle that finds another turhtle nearby
            always moves to its cell. A social affinity of -1 means that a turtle always tries
            to avoid any turtle nearby.
            nd, a parameter that decides the number of doors (largest for nd=1)
        '''

        # read the map
        self.map_bt                 = np.genfromtxt(map_file)
        self.social_affinity        = social_affinity
        self.avoid_awareness        = -social_affinity

        if print_level(prtl, PrtLvl.Concise):
            print(f'loaded barrio tortuga map with dimensions ->{ self.map_bt.shape}')
            if self.social_affinity >= 0:
                print(f'social affinity ->{ self.social_affinity}')
            else:
                print(f'avoid awareness ->{ self.avoid_awareness}')


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

        for i in range(int(self.turtles)):
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


    def get_doors(self, nd):
        l,w = self.map_bt.shape
        D = []
        if nd == 1:
            return [(x,y) for x in range(l) for y in range(w) if is_house(self.map_bt, x, y) == False and is_house(self.map_bt, x, y-1) == True]
        else:
            return [(x,y) for x in range(l) for y in range(w) if is_house(self.map_bt, x, y) == False and is_house(self.map_bt, x-1, y-1) == True]
