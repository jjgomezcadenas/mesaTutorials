from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import UserSettableParameter

from barrio_tortuga.BarrioTortugaSEIR import BarrioTortugaSEIR
from barrio_tortuga.BarrioTortugaSEIR import SeirTurtle

def turtle_portrayal(agent):
    portrayal = {}
    if type(agent) is SeirTurtle:
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "r": 0.5}
        if agent.kind == "S":
            portrayal["Color"] = "white"
            portrayal["Layer"] = 0
        elif agent.kind == "E":
            portrayal["Color"] = "blue"
            portrayal["Layer"] = 0
        elif agent.kind == "I":
            portrayal["Color"] = "red"
            portrayal["Layer"] = 0
        elif agent.kind == "B":
            portrayal["Color"] = "green"
            portrayal["Layer"] = 0

    return portrayal


canvas_element = CanvasGrid(turtle_portrayal, 20, 20, 400, 400)
chart          = ChartModule([{"Label": 'NumberOfContacts', "Color": "#0000FF"}]
)

model_params = {"turtles": UserSettableParameter('slider', 'turtles', 100, 100, 5000, 100),
                "i0" : UserSettableParameter('slider', 'i0', 10, 1, 100, 5),
                "r0" : UserSettableParameter('slider', 'r0', 3.5, 0.5, 10.5, 0.5),
                "ti" : UserSettableParameter('slider', 'ti', 5, 1, 20, 1),
                "tr" : UserSettableParameter('slider', 'ti', 5, 1, 20, 1)}


server = ModularServer(BarrioTortugaSEIR, [canvas_element, chart], "Barrio Tortuga SEIR",
model_params)
server.port = 8521
server.launch()
