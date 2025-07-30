import fastf1
import pandas as pd
import os

# os.makedirs('cache', exist_ok=True)
# # q = os.getcwd()
# # print(q)

fastf1.Cache.enable_cache('cache')

def get_fastest_lap( year: int, gp: list, drivers: list, mode: str):
    for i in gp:
        try:
            session = fastf1.get_session(year, i, mode)
            session.load()

            for j in drivers:
                driver_lap = session.laps.pick_drivers(j).pick_fastest()
                driver_tel = driver_lap.get_car_data()

                print(f'GP- {i};Driver - {j}')

                try:
                    match mode:
                        case 'Q':
                            path = f"new-telemetry/Qualifying/{i}-quali-{j}.csv"
                        case 'R':
                            path = f"new-telemetry/Race/{i}-race-{j}.csv"
                except:
                    print(f"Path issue")

                driver_tel = driver_tel.drop(labels=["Date", "Time", "SessionTime"], axis=1)
                driver_tel.to_csv(path, index=False)

        except:
            print(f"Data not found for {i}")

races = ['Australia', 'China', 'Japan', 'Bahrain']
drivers = ['VER', 'TSU', 'NOR', 'PIA', 'RUS', 'ANT', 'HAM', 'LEC', 'SAI', 'ALB']
year = 2025

get_fastest_lap(year, races, drivers, 'Q')
# get_fastest_lap(year, races, drivers, 'R')
