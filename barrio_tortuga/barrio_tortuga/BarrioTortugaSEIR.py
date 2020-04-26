from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa import Agent
from mesa.time import RandomActivation
import numpy as np

from . utils import PrtLvl, print_level, throw_dice

CALIB = True  # calib  : set True to calibrate the model.
NC = 2.3 # number of contacts (found from calibration)
TYPES = ['Susceptible','Exposed', 'Infected', 'Recovered']
NCS ={1000:2.3, 5000:23.9}

prtl=PrtLvl.Concise

def number_turtles_in_cell(cell):
    turtles = [obj for obj in cell if isinstance(obj, SeirTurtle)]
    return len(turtles)


def turtles_in_cell(cell):
    if number_turtles_in_cell(cell) > 0:
        return True
    else:
        return False


def number_of_infected(model):
    a =[agent for agent in model.schedule.agents if agent.kind=='I']
    return len(a)


def number_of_susceptible(model):
    a =[agent for agent in model.schedule.agents if agent.kind=='S']
    return len(a)


def number_of_recovered(model):
    a =[agent for agent in model.schedule.agents if agent.kind=='R']
    return len(a)


def number_of_exposed(model):
    a =[agent for agent in model.schedule.agents if agent.kind=='E']
    return len(a)


def number_of_contacts(model):
    nc = 0
    ng = 0
    for y in range(model.grid.height):
        for x in range(model.grid.width):
            ng += 1
            this_cell = model.grid.get_cell_list_contents((x,y))
            n_turtles = number_turtles_in_cell(this_cell)
            if n_turtles > 0:
                nc += n_turtles -1
    return nc / ng

def in_range(x, xmin, xmax):
    if x >= xmin and x < xmax:
        return True
    else:
        return False

def number_of_turtles_in_neighborhood(model):
    nc = 0
    ng = 0
    NC = []
    for y in range(model.grid.height):
        for x in range(model.grid.width):
            ng +=1

            if print_level(prtl, PrtLvl.Verbose):
                if in_range(x,2,3) and in_range(y,2,3):
                    print(f'x = {x} y = {y}')

            c = model.grid.get_cell_list_contents((x,y))
            n_turtles_in_c = number_turtles_in_cell(c)

            if print_level(prtl, PrtLvl.Verbose):
                if in_range(x,0,3) and in_range(y,2,3):
                    print(f'number of turtles in this cell  = {n_turtles_in_c}')

            if n_turtles_in_c > 0:
                #coordinates of neighbors
                n_xy = model.grid.get_neighborhood((x,y), model.moore, True)

                if print_level(prtl, PrtLvl.Verbose):
                    if in_range(x,2,3) and in_range(y,2,3):
                        print(f'coordinates of neighbors inlcuding center = {n_xy}')

                ncc = 0
                for xy in n_xy:
                    if print_level(prtl, PrtLvl.Verbose):
                        if in_range(x,2,3) and in_range(y,2,3):
                            print(f'coordinates of neighbors = {xy}')

                    cn = model.grid.get_cell_list_contents(xy)
                    n_turtles_nb = number_turtles_in_cell(cn)

                    if print_level(prtl, PrtLvl.Verbose):
                        if in_range(x,2,3) and in_range(y,2,3):
                            print(f'nof turtles = {n_turtles_nb}')

                    if n_turtles_nb > 0:
                        nc += n_turtles_nb
                        ncc += n_turtles_nb
                NC.append(ncc)

                if print_level(prtl, PrtLvl.Verbose):
                    if in_range(x,2,3) and in_range(y,2,3):
                        print(f'nof turtles in cell and neighbors = {ncc}')

    if print_level(prtl, PrtLvl.Verbose):
        print(f'NC = {NC}')
        print(f'NC mean = {np.mean(NC)}')
    #return (nc -1) /ng
    return np.mean(NC)


class BarrioTortugaSEIR(Model):
    """A simple model of SEIR epidemics.

    The parameters are:
        turtles: the number of agents in the simulation.
        i0     : the initial number of infected agents.
        r0     : basic reproductive number
        p      : transmission probability per contact
        ti     : incubation time
        tr     : recovery time = duration of infectiouness.


        Model calibration. R0 can be written as:

        R0 = c x p x ti

        where:
           c is the average number of contacts per unit time
           p is the transmission probability per contact
           ti is the duration of the infection (the time the turtle is in the I cathegory)

       Two quantities can be taken as known, R0 and ti. Then one can determine p as:
           p = R0 /(c * ti)

        To find c we calibrate the model, that is we run the simulation without infection and count the number of contacts per unit time.

    """

    def __init__(self, turtles=1000, i0=10, r0 = 3.5, nc = 2.3,
                 ti = 5, tr = 5, width=40, height=40, calib = False,
                 prtl=PrtLvl.Concise):
        self.turtles = turtles
        self.i0      = i0
        self.r0      = r0
        self.ti      = ti
        self.tr      = tr
        self.tick    = 0
        self.calib   = calib

        if self.calib:
            self.p       = 1
        else:
            self.p   = self.r0 /(nc * self.tr)
        #self.p       = p
        self.fi      = i0 / turtles
        self.height  = height
        self.width   = width
        self.grid                   = MultiGrid(self.height, self.width, torus=True)
        self.moore                  = True
        self.schedule               = RandomActivation(self)

        if self.calib:
            self.datacollector          = DataCollector(
            model_reporters             = {"NumberOfneighbors": number_of_turtles_in_neighborhood}

            )
        else:
            self.datacollector          = DataCollector(
            model_reporters             = {"NumberOfInfected": number_of_infected,
                                           "NumberOfSusceptible": number_of_susceptible,
                                           "NumberOfRecovered": number_of_recovered,
                                           "NumberOfExposed": number_of_exposed}
            )


        if print_level(prtl, PrtLvl.Concise):
            print(f"""Barrio Tortuga SEIR parameters
            number of turtles       = {self.turtles}
            initial infected        = {self.i0}
            fraction infected       = {self.fi}
            R0                      = {self.r0}
            Ti (incubation time)    = {self.ti}
            Tr (recovery time)      = {self.tr}
            number of contacts      = {nc}
            P (trans. prob/contact) = {self.p}
            Calib                   = {self.calib}
            Grid (w x h)            = {self.width} x {self.height}
            """)


        # Create agents
        if self.calib:
            for i in range(self.turtles):
                x,y = self.random_pos()           # random position
                a = SeirTurtle(i, (x, y), 'S', ti, tr, self, True)
                self.schedule.add(a)              # add to schedule
                self.grid.place_agent(a, (x, y))  # added to schedule

        else:
            ss = self.turtles - i0
            A = ss * ['S'] + i0 * ['I']
            #print(f'A = {A}')
            np.random.shuffle(A)
            #print(f'A (rd) = {A}')

            for i, at in enumerate(A):
                x,y = self.random_pos()           # random position

                if at == 'I':
                    a = SeirTurtle(i, (x, y), 'I', ti, tr, self, True)
                else:
                    a = SeirTurtle(i, (x, y), 'S', ti, tr, self, True)

                self.schedule.add(a)              # add to schedule
                self.grid.place_agent(a, (x, y))  # added to schedule


        self.running = True
        self.datacollector.collect(self)

    def step(self):
        self.tick +=1                      # step global clock
        self.schedule.step()               # step all turtles
        self.datacollector.collect(self)

    def random_pos(self):
        x = self.random.randrange(self.width)
        y = self.random.randrange(self.height)
        return x, y

    def number_of_agents(self, kind):
        if kind == 'A':
            a =[agent for agent in self.schedule.agents]
        else:
            a =[agent for agent in self.schedule.agents if agent.kind==kind]
        return len(a)

    @property
    def new_infected(self):
        a =np.array([turtle.nb_infected for turtle in self.schedule.agents])
        return np.sum(a)

    @property
    def new_recovered(self):
        a =np.array([turtle.nb_recovered for turtle in self.schedule.agents])
        return np.sum(a)

    @property
    def new_exposed(self):
        a =np.array([turtle.nb_exposed for turtle in self.schedule.agents])
        return np.sum(a)


class SeirTurtle(Agent):
    '''
    Class implementing a SEIR turtle that can move at random

    '''

    def __init__(self, unique_id, pos, kind, ti, tr, model, moore=True):
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
        self.kind = kind
        self.ti = ti           # equals model average for now throw dist later
        self.tr = tr           # equals model average for now throw dist later
        self.il = 0            # counter tick
        self.iil = 0           # infection length
        self.iel = 0           # infection length


    def step(self):
        self.il +=1  # increment counter

        if self.kind == 'E' :
            self.iel += 1
            if self.iel > self.ti :  # time of infection larger than incubation
                self.kind = 'I'
                self.iil = 0

        elif self.kind == 'I' :     #infected can infect
            self.iil +=1
            self.infect()
            if self.iil > self.tr :
                self.kind = 'R'

        self.random_move()


    def infect(self):
        n_xy = self.model.grid.get_neighborhood(self.pos, self.model.moore, True)

        if print_level(prtl, PrtLvl.Verbose):
                print(f'coordinates of neighbors, including me = {n_xy}')

        for xy in n_xy:
            if print_level(prtl, PrtLvl.Verbose):
                    print(f'neighbors = {xy}')

            cell = self.model.grid.get_cell_list_contents(xy)
            turtles = [obj for obj in cell if isinstance(obj, type(self))]

            if print_level(prtl, PrtLvl.Verbose):
                print(f' number of turtles = {len(turtles)}')

            for turtle in turtles:

                if print_level(prtl, PrtLvl.Verbose):
                    print(f' turtle kind = {turtle.kind}')

                if turtle.kind == 'S':
                    if print_level(prtl, PrtLvl.Verbose):
                        print(f' throwing dice')
                    if throw_dice(self.model.p):

                        if print_level(prtl, PrtLvl.Verbose):
                            print(f' turning turtle into E')

                        turtle.kind = 'E'


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
