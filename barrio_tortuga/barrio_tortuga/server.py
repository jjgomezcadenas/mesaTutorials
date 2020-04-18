from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import UserSettableParameter

from barrio_tortuga.BarrioTortuga import BarrioTortuga

def turtle_portrayal(agent):
    if agent is None:
        return

    portrayal = {}

    portrayal["Shape"] = "barrio_tortuga/resources/ninja-turtle.png"
        # https://icons8.com/icons/set/turtle
    portrayal["scale"] = 0.5
    portrayal["Layer"] = 1

    return portrayal


canvas_element = CanvasGrid(turtle_portrayal, 50, 50, 1000, 1000)


model_params = {"turtles": UserSettableParameter('slider', 'turtles', 100, 100, 5000)}


server = ModularServer(BarrioTortuga, [canvas_element], "Barrio Tortuga", model_params)
server.port = 8521
