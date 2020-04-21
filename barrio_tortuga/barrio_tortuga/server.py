from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import UserSettableParameter

from barrio_tortuga.BarrioTortuga import BarrioTortuga, Turtle, Patch


def turtle_portrayal(agent):
    if agent is None:
        return

    portrayal = {}
    if type(agent) is Turtle:
        portrayal["Shape"] = "barrio_tortuga/resources/ninja-turtle.png"
            # https://icons8.com/icons/set/turtle
        portrayal["scale"] = 0.9
        portrayal["Layer"] = 1

    elif type(agent) is Patch:
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "r": 0.5}
        if agent.kind == 1:
            portrayal["Color"] = "red"
            portrayal["Layer"] = 0
        else:
            portrayal["Color"] = "grey"
            portrayal["Layer"] = 1
            portrayal["r"] = 0.2

    return portrayal


canvas_element = CanvasGrid(turtle_portrayal, 100, 100, 1000, 1000)
chart          = ChartModule([{"Label": 'NumberOfEncounters', "Color": "#0000FF"}]
)

model_params = {"turtles": UserSettableParameter('slider', 'turtles', 100, 10, 500),
                "social_affinity" : UserSettableParameter('slider', 'avoid_awareness', 0.,
                -1., 1., 0.1)}

server = ModularServer(BarrioTortuga, [canvas_element, chart], "Barrio Tortuga", model_params)
server.port = 8521
