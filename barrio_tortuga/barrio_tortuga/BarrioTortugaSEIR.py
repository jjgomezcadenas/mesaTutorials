from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa import Agent
from mesa.time import RandomActivation
import numpy as np

from scipy.stats import gamma
from . stats import c19_weib_rvs

from . utils import PrtLvl, print_level, throw_dice

CALIB = False
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
           c = nc x N /a
           where, for large density :
           nc = 9 (number of cells in Moore)
           N  = number of turtles
           a  = area of grid (w x h)

           p is the transmission probability per contact
           ti is the duration of the infection (the time the turtle is in the I cathegory)

       Two quantities can be taken as known, R0 and ti. Then one can determine p as:
           p = R0 /(c * ti)

    """


    def __init__(self,
                 ticks_per_day =    5,
                 turtles       = 1000,
                 i0            =   10,
                 r0            =    3.5,
                 ti_shape      =    5.8,
                 ti_scale      =    0.95,
                 tr            =    5,
                 tr_rms        =    2.6,
                 width         =   40,
                 height        =   40,
                 stochastic    =  False):

        self.moore    = True       # grid includes all surroundng neighbors
        self.ticks_per_day = ticks_per_day
        self.stoc     = stochastic # if False, use mean values, otherwise draw from stats dist
        self.turtles  = turtles    # Total population
        self.i0       = i0         # initial number of infected
        self.r0       = r0         # basic reproductive number
        self.tr       = tr         # recovery time
        self.tr_rms   = tr_rms
        self.ti_shape = ti_shape   # incubation time shape in gamma distribution
        self.ti_scale = ti_scale   # incubation time scale in gamma distribution
        self.stochastic = stochastic # True if stochastic run

        # ti (average) and ti_var (variance) obtained from gamma distribution
        self.ti, self.ti_var  = gamma.stats(a=ti_shape, scale=ti_scale, moments='mv')
        #self.ti = 5.0
        # average number of contacts:  nc = 9 * N / area
        # where N / area is the average population per cell and 9 the number of cells
        self.nc      = 9 * turtles / (width * height) # 9 -> 9 cells in moore

        # CALIB can be used in the case of a non-homogeneous population
        if CALIB:
            self.p       = 1
        else:
            self.p   = self.r0 /(self.nc * self.tr * ticks_per_day)

        # Grid and schedule
        self.height     = height
        self.width      = width
        self.grid       = MultiGrid(self.height, self.width, torus=True)

        self.schedule   = RandomActivation(self)

        if CALIB:
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
            if CALIB:
                print(f"Running in CALIB mode")

            print(f"""Barrio Tortuga SEIR parameters
            Stochastic simulation?  = {self.stochastic}
            number of turtles       = {self.turtles}
            initial infected        = {self.i0}
            R0                      = {self.r0}
            Ti (incubation time)    = {self.ti}
            Ti (shape)              = {self.ti_shape}
            Ti (scale)              = {self.ti_scale}
            Tr (recovery time)      = {self.tr}
            number of contacts      = {self.nc}
            P (trans. prob/contact) = {self.p}
            Grid (w x h)            = {self.width} x {self.height}
            """)


        # Create agents
        if CALIB:  # only susceptible agents
            for i in range(self.turtles):
                x,y = self.random_pos()           # random position
                a = SeirTurtle(i, (x, y), 'S', self.ti, self.tr, self)
                self.schedule.add(a)              # add to schedule
                self.grid.place_agent(a, (x, y))  # added to schedule

        else:
            ss = self.turtles - i0            # number of susceptibles
            A = ss * ['S'] + i0 * ['I']
            np.random.shuffle(A)              # in random order


            for i, at in enumerate(A):
                x,y = self.random_pos()           # random position

                if at == 'I':
                    a = SeirTurtle(i, (x, y), 'I',
                                   self.ti * ticks_per_day,
                                   self.tr * ticks_per_day,
                                   self)
                else:
                    if stochastic:  # generate ti according to gamma
                        ti = gamma.rvs(a=ti_shape, scale=ti_scale)
                        if print_level(prtl, PrtLvl.Concise):
                            if i < 5:
                                print (f' creating turtle with ti ={ti}')
                        a = SeirTurtle(i, (x, y), 'S',
                                       ti * ticks_per_day,
                                       self.tr * ticks_per_day,
                                       self)
                    else:
                        a = SeirTurtle(i, (x, y), 'S',
                                       self.ti * ticks_per_day,
                                       self.tr * ticks_per_day,
                                       self)

                self.schedule.add(a)              # add to schedule
                self.grid.place_agent(a, (x, y))  # added to schedule


        self.running = True
        self.datacollector.collect(self)

    def step(self):
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

    def __init__(self, unique_id, pos, kind, ti, tr, model):
        '''
        grid: The MultiGrid object in which the agent lives.
        x: The agent's current x coordinate
        y: The agent's current y coordinate
        stochastic: If false mean average
        '''
        super().__init__(unique_id, model)
        self.pos = pos
        self.kind = kind
        self.ti = ti           # equals model average for now throw dist later
        self.tr = tr           # equals model average for now throw dist later
        self.il = 0            # counter tick
        self.iil = 0           # infection length
        self.iel = 0           # infection length


    def step(self):
        self.il+=1

        # Turtle became exposed with tag self.iel (see infect ())
        if self.kind == 'E' :
        #and self.model.schedule.steps > self.iel:

            if print_level(prtl, PrtLvl.Detailed):
                    print(f"""Found exposed with tag = {self.iel}
                          global time = {self.model.schedule.steps}
                          turtle id   = {self.unique_id}
                """)

            # When time is larger than incubation time, become infected
            if self.model.schedule.steps - self.iel > self.ti :
                self.iil = self.model.schedule.steps
                self.kind = 'I'

                if print_level(prtl, PrtLvl.Detailed):
                    print(f"""Turning E into I with tag = {self.iil}
                          global time = {self.model.schedule.steps}
                          turtle id   = {self.unique_id}
                """)

        elif self.kind == 'I':
            self.infect()

            if print_level(prtl, PrtLvl.Detailed):
                print(f"""Found Infected with tag = {self.iil}
                          global time = {self.model.schedule.steps}
                          turtle id   = {self.unique_id}
                          **going to infect***
                """)

            # When time is larger than recovery time, become recovered
            if self.model.schedule.steps - self.iil >  self.tr :
                self.kind = 'R'

                if print_level(prtl, PrtLvl.Detailed):
                    print(f"""Turning I into R with tag = {self.iil}
                          global time = {self.model.schedule.steps}
                          turtle id   = {self.unique_id}
                """)
            # else:
            #     self.infect()
            #
            #     if print_level(prtl, PrtLvl.Detailed):
            #         print(f"""Found Infected with tag = {self.iil}
            #                   global time = {self.model.schedule.steps}
            #                   turtle id   = {self.unique_id}
            #                   **going to infect***
            #         """)



        self.random_move()


    def infect(self):
        if print_level(prtl, PrtLvl.Verbose):
                print(f"""Now infecting with tags  {self.iil}
                          global time = {self.model.schedule.steps}
                          turtle id   = {self.unique_id}

                """)
        # array of coordinates ((x0,y0), (x1,y1), (x2, y2),...) of neighbors
        # last parameter True inludes own cell
        n_xy = self.model.grid.get_neighborhood(self.pos, self.model.moore, True)

        if print_level(prtl, PrtLvl.Verbose):
                print(f'coordinates of neighbors, including me = {n_xy}')

        for xy in n_xy:   # loops over all cells
            if print_level(prtl, PrtLvl.Verbose):
                    print(f'neighbors = {xy}')

            cell = self.model.grid.get_cell_list_contents(xy)
            turtles = [obj for obj in cell if isinstance(obj, type(self))]

            if print_level(prtl, PrtLvl.Verbose):
                print(f' number of turtles = {len(turtles)}')

            for turtle in turtles:  # loops over all turtles in cells

                if print_level(prtl, PrtLvl.Verbose):
                    print(f' turtle kind = {turtle.kind}')

                if turtle.kind == 'S':  # if susceptible found try to infect
                    if print_level(prtl, PrtLvl.Verbose):
                        print(f' throwing dice')

                    if throw_dice(self.model.p):
                        turtle.kind = 'E'
                        turtle.iel = self.model.schedule.steps # tag = infection time

                        if print_level(prtl, PrtLvl.Detailed):
                            print(f' **TURNING TURTLE INTO E ** ')
                            print(f"""tag  {turtle.iel}
                                      global time = {self.model.schedule.steps}
                                      turtle id   = {turtle.unique_id}

                            """)

    def random_move(self):
        '''
        Step one cell in any allowable direction.
        We use grid’s built-in get_neighborhood method, which returns all the neighbors of a given cell.
        This method can get two types of cell neighborhoods: Moore (including diagonals),
        and Von Neumann (only up/down/left/right).
        It also needs an argument as to whether to include the center cell itself as one of the neighbors.
        '''
        # Pick the next cell from the adjacent cells.
        next_moves = self.model.grid.get_neighborhood(self.pos, self.model.moore, True)
        next_move = self.random.choice(next_moves)
        # Now move:
        self.model.grid.move_agent(self, next_move)
