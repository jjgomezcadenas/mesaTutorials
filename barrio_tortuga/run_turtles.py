from scipy.stats import gamma
from barrio_tortuga.BarrioTortugaSEIR import BarrioTortugaSEIR
import pandas as pd
import os
import sys
import shutil



def run_turtles(steps          = 500,
                fprint         = 25,
                ticks_per_day  = 5,
                turtles        = 10000,
                i0             = 10,
                r0             = 3.5,
                ti             = 5.5,
                tr             =  5.5,
                ti_dist        = 'F',    # F for fixed, E for exp G for Gamma
                tr_dist        = 'F',
                p_dist         = 'F',    # F for fixed, S for Binomial, P for Poissoin
                width          = 40,
                height         = 40):

    print(f" Running Simulation with {turtles}  turtles, for {steps} steps.")
    bt = BarrioTortugaSEIR(ticks_per_day, turtles, i0, r0, ti, tr,
                           ti_dist, tr_dist, p_dist,
                           width, height)

    for i in range(steps):
        if i%fprint == 0:
            print(f' step {i}')
        bt.step()
    print('Done!')

    STATS = {}
    STATS['Ti'] = bt.Ti
    STATS['Tr'] = bt.Tr
    STATS['P']  = bt.P
    return bt.datacollector.get_model_vars_dataframe(), pd.DataFrame.from_dict(STATS)

# Directory
directory = "GeeksForGeeks"



def run_series(ns=100,
               csv            = False,
               path           ="/Users/jjgomezcadenas/Projects/Development/mesaTutorials/data",
               steps          = 500,
               fprint         = 25,
               ticks_per_day  = 5,
               turtles        = 10000,
               i0             = 10,
               r0             = 3.5,
               ti             = 5.5,
               tr             =  5.5,
               ti_dist        = 'F',    # F for fixed, E for exp G for Gamma
               tr_dist        = 'F',
               p_dist         = 'F',    # F for fixed, S for Binomial, P for Poissoin
               width          = 40,
               height         = 40):

    if csv:
        fn1 = f'Turtles_{turtles}_steps_{steps}_i0_{i0}_r0_{r0}_ti_{ti}_tr_{tr}'
        fn2 = f'Tid_{ti_dist}_Tir_{tr_dist}_Pdist_{p_dist}'
        dirname =f'{fn1}_{fn2}'
        mdir = os.path.join(path, dirname)

        try:
            shutil.rmtree(mdir, ignore_errors=False, onerror=None)
            print(f"Directory {mdir} has been removed" )
        except OSError as error:
            print(error)
            print("Directory {mdir} not removed")

        print(f" Creating Directory {mdir} created")
        try:
            os.mkdir(mdir)
            print(f"Directory {mdir} has been created" )
        except OSError as error:
            print(error)
            print("Directory {mdir} not created")
            sys.exit()


    STATS = []
    DFT   = []
    for i in range(ns):
        dft, stats = run_turtles(steps, fprint, ticks_per_day, turtles, i0, r0,
                                 ti, tr,
                                 ti_dist, tr_dist, p_dist,
                                 width, height)

        STATS.append(stats)
        DFT.append(dft)


    df  = pd.concat(DFT)
    dfs = pd.concat(STATS)

    if csv:

        for i in range(len(DFT)):
            file =f'DFT_run_{i}.csv'
            mfile = os.path.join(mdir, file)
            DFT[i].to_csv(mfile, sep=" ")

        file=f'DFT_run_average.csv'
        mfile = os.path.join(mdir, file)
        df.groupby(df.index).mean().to_csv(mfile, sep=" ")

        file=f'STA.csv'
        mfile = os.path.join(mdir, file)
        dfs.groupby(dfs.index).mean().to_csv(mfile, sep=" ")

run_series(ns             = 2,
           csv            = True,
           steps          = 500,
           fprint         = 25,
           ticks_per_day  = 5,
           turtles        = 10000,
           i0             = 10,
           r0             = 3.5,
           ti             = 5.5,
           tr             = 8.5,
           ti_dist        = 'F',    # F for fixed, E for exp G for Gamma
           tr_dist        = 'F',
           p_dist         = 'F',    # F for fixed, S for Binomial, P for Poissoin
           width          = 40,
           height         = 40)
