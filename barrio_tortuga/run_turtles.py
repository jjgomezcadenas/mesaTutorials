from scipy.stats import gamma
from barrio_tortuga.BarrioTortugaSEIR import BarrioTortugaSEIR
import pandas as pd

def run_turtles(steps          = 500,
                fprint         = 25,
                ticks_per_day  = 5,
                turtles        = 10000,
                i0             = 10,
                r0             = 3.5,
                ti_shape       = 5.8,
                ti_scale       = 0.95,
                tr             = 5,
                tr_rms         = 2.6,
                width          = 40,
                height         = 40,
                stochastic     = False ):

    print(f" Running Barrio Tortuga SEIR with {turtles}  turtles, for {steps} steps. Stochastic = {stochastic}")
    bt = BarrioTortugaSEIR(ticks_per_day, turtles, i0, r0, ti_shape, ti_scale, tr, tr_rms, width, height, stochastic)

    for i in range(steps):
        if i%fprint == 0:
            print(f' step {i}')
        bt.step()
    print('Done!')
    return bt.datacollector.get_model_vars_dataframe()


def run_series(ns=100,
               csv            = False,
               steps          = 500,
               fprint         = 25,
               ticks_per_day  = 5,
               turtles        = 10000,
               i0             = 10,
               r0             = 3.5,
               ti_shape       = 5.8,
               ti_scale       = 0.95,
               tr             = 5,
               tr_rms         = 2.6,
               width          = 40,
               height         = 40,
               stochastic     = False):
    DFT = [run_turtles(steps, fprint, ticks_per_day, turtles, i0, r0, ti_shape, ti_scale, tr, tr_rms, width, height, stochastic) for i in range(ns)]
    df = pd.concat(DFT)

    if stochastic:
        stoc = 'S'
    else:
        stoc = 'F'
    if csv:
        ti, _  = gamma.stats(a=ti_shape, scale=ti_scale, moments='mv')
        for i in range(len(DFT)):
            file=f'{stoc}_Turtles_n_{turtles}_i0_{i0}_r0_{r0}_ti_{ti}_tr_{tr}_run_{i}.csv'
            DFT[i].to_csv(file, sep=" ")

        file=f'{stoc}_Turtles_n_{turtles}_i0_{i0}_r0_{r0}_ti_{ti}_tr_{tr}_run_average.csv'
        df.groupby(df.index).mean().to_csv(file, sep=" ")

    return df.groupby(df.index).mean()

#ti_shape       = 5.8,
dft10kS25 = run_series(ns=10,
               csv            = True,
               steps          = 1000,
               fprint         = 100,
               ticks_per_day  = 10,
               turtles        = 10000,
               i0             = 10,
               r0             = 3.5,
               ti_shape       = 5.8,
               ti_scale       = 0.95,
               tr             = 5,
               tr_rms         = 2.6,
               width          = 40,
               height         = 40,
               stochastic     = True)
