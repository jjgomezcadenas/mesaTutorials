from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa import Agent
from mesa.time import RandomActivation
import numpy as np

from scipy.stats import gamma
from scipy.stats import expon
from . stats import c19_nbinom_rvs

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


def get_time(t_dist, t_mean):
    if t_dist == 'E':
        return expon.rvs(scale=t_mean)
    elif t_dist == 'G':
        return gamma.rvs(a=t_mean, scale=1.0)
    else:
        return t_mean


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

        Stochastic behaviour can be introduced in:
        1) ti and tr, which can be chosen to be either gamma or exponentially distributed
        2) p, via R0, which can be chosen to be either negative binomial or Poisson.

    """


    def __init__(self,
                 ticks_per_day =    5,
                 turtles       = 1000,
                 i0            =   10,
                 r0            =    3.5,
                 ti            =    5.5,
                 tr            =    3.5,
                 ti_dist       =    'F',    # F for fixed, E for exp G for Gamma
                 tr_dist       =    'F',
                 p_dist        =    'F',    # F for fixed, S for Binomial, P for Poissoin
                 width         =   40,
                 height        =   40):


        # define grid and schedule
        self.ticks_per_day = ticks_per_day
        self.height     = height
        self.width      = width
        self.grid       = MultiGrid(self.height, self.width, torus=True)
        self.moore      = True
        self.schedule   = RandomActivation(self)

        self.turtles  = turtles
        self.i0       = i0
        self.r0       = r0

        # average number of contacts:  nc = 9 * N / area
        # where N / area is the average population per cell and 9 the number of cells
        self.nc         = 9 * self.turtles / (self.width * self.height)

        # counters
        self.P        = []
        self.Ti       = []
        self.Tr       = []

        # distribution types
        self.ti_dist = ti_dist
        self.tr_dist = tr_dist
        self.p_dist  = p_dist

        self.ti = ti
        self.tr = tr

        # ti (average) and ti_var (variance)
        # if   self.ti_dist == 'E':
        #     self.ti, self.ti_var  = expon.stats(scale=ti, moments='mv')
        # elif self.ti_dist == 'G':
        #     self.ti, self.ti_var  = gamma.stats(a=ti_shape, scale=1, moments='mv')
        # else:
        #     self.ti = ti
        #     self.ti_var = 0

        # te (average) and ti_var (variance)
        # if   self.tr_dist == 'E':
        #     self.tr, self.tr_var  = expon.stats(scale=tr, moments='mv')
        # elif self.tr_dist == 'G':
        #     self.tr, self.tr_var  = gamma.stats(a=tr_shape, scale=tr_scale, moments='mv')
        # else:
        #     self.tr = tr
        #     self.tr_var = 0

        # Prob
        self.k = 1
        if   self.p_dist == 'S':
            self.k  = 0.16
        elif self.p_dist == 'P':
            self.k = 1e+4

        # infection probability for fixed case
        self.p   = self.r0 /(self.nc * self.tr * self.ticks_per_day)

        if print_level(prtl, PrtLvl.Concise):
            print(f""" Simulation Parameters:

            General
                number of turtles       = {self.turtles}
                initial infected        = {self.i0}
                ticks per day           = {self.ticks_per_day}
                Grid (w x h)            = {self.width} x {self.height}

            Control of stochastics

                ti_dist = {ti_dist}
                tr_dist = {tr_dist}
                p_dist =  {ti_dist}

            Average parameters controlling t and prob

                ti     =  {self.ti}
                tr     =  {self.tr}
                k      =  {self.k}
                r0     =  {self.r0}
                nc     =  {self.nc}
                p      =  {self.p}

            """)

        # Data collector
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


        # Create turtles
        if CALIB:  # only susceptible agents
            for i in range(self.turtles):
                x,y = self.random_pos()           # random position
                a = SeirTurtle(i, (x, y), 'S', ti, tr, 1, self)
                self.schedule.add(a)              # add to schedule
                self.grid.place_agent(a, (x, y))  # added to schedule

        else:
            ss = self.turtles - i0            # number of susceptibles
            A = ss * ['S'] + i0 * ['I']
            np.random.shuffle(A)              # in random order

            for i, at in enumerate(A):
                x,y = self.random_pos()           # random position
                ti = get_time(self.ti_dist, self.ti)
                tr = get_time(self.tr_dist, self.tr)
                p  = self.get_prob()
                self.Ti.append(ti)
                self.Tr.append(tr)
                self.P.append(p)

                if print_level(prtl, PrtLvl.Concise):
                    if i < 5:
                        print (f' creating turtle number {i} with ti = {ti}, tr = {tr}, p ={p:.2e}')

                if at == 'I':
                    if print_level(prtl, PrtLvl.Concise):
                        if i < 5:
                            print (f' creating I turtle')

                    a = SeirTurtle(i, (x, y), 'I',
                                   ti * ticks_per_day,
                                   tr * ticks_per_day, p,
                                   self)

                else:
                    if print_level(prtl, PrtLvl.Concise):
                        if i < 5:
                            print (f' creating S turtle')

                    a = SeirTurtle(i, (x, y), 'S',
                                   ti * ticks_per_day,
                                   tr * ticks_per_day, p,
                                   self)

                self.schedule.add(a)              # add to schedule
                self.grid.place_agent(a, (x, y))  # added to schedule

        self.running = True
        self.datacollector.collect(self)


    def get_prob(self):
        if self.p_dist == 'S' or self.p_dist == 'P':
            r0  = c19_nbinom_rvs(self.r0, self.k) # self.k decided which one
            p   = r0 /(self.nc * self.tr * self.ticks_per_day)
        else:
            p = self.p
        return p


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


class SeirTurtle(Agent):
    '''
    Class implementing a SEIR turtle that can move at random

    '''

    def __init__(self, unique_id, pos, kind, ti, tr, prob, model):
        '''
        grid: The MultiGrid object in which the agent lives.
        x: The agent's current x coordinate
        y: The agent's current y coordinate
        stochastic: If false mean average
        '''
        super().__init__(unique_id, model)
        self.pos  = pos
        self.p    = prob
        self.kind = kind
        self.ti   = ti           # equals model average for now throw dist later
        self.tr   = tr           # equals model average for now throw dist later
        self.il   = 0            # counter tick
        self.iil  = 0           # infection length
        self.iel  = 0           # infection length


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

                    if throw_dice(self.p):
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
