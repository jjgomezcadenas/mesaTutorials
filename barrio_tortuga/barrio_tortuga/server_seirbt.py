from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import UserSettableParameter

from barrio_tortuga.BarrioTortugaSEIR import BarrioTortugaSEIR
from barrio_tortuga.BarrioTortugaSEIR import SeirTurtle
from barrio_tortuga.BarrioTortugaSEIR import CALIB

def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "r": 0.2}

    if agent.kind == "S":
        portrayal["Color"] = "grey"
        portrayal["Layer"] = 0
    elif agent.kind == "E":
        portrayal["Color"] = "blue"
        portrayal["Layer"] = 0
    elif agent.kind == "I":
        portrayal["Color"] = "red"
        portrayal["Layer"] = 0
    elif agent.kind == "R":
        portrayal["Color"] = "green"
        portrayal["Layer"] = 0

    return portrayal


canvas_element = CanvasGrid(agent_portrayal, 40, 40, 800, 800)
if CALIB:
    chart          = ChartModule([{"Label" : "NumberOfneighbors", "Color": "#666666"}]
                              )
else:
    chart          = ChartModule([{"Label": 'NumberOfInfected', "Color": "#AA0000"},
                                  {"Label": 'NumberOfSusceptible', "Color": " #AA5500"},
                                  {"Label": 'NumberOfRecovered', "Color": "#00AA00"},
                                  {"Label": 'NumberOfExposed', "Color": "#0000AA"}]
                              )

model_params = {"turtles": UserSettableParameter('slider', 'turtles', 1000, 100, 20000, 100),
                "i0" : UserSettableParameter('slider', 'i0', 10, 1, 100, 5),
                "r0" : UserSettableParameter('slider', 'r0', 3.5, 0.5, 10.5, 0.5),
                "ti_shape" : UserSettableParameter('slider', 'ti', 5.8, 1, 20, 1),
                "tr" : UserSettableParameter('slider', 'tr', 5, 1, 20, 1)}


server = ModularServer(BarrioTortugaSEIR, [canvas_element, chart], "Barrio Tortuga SEIR",
model_params)
server.port = 8521
server.launch()
