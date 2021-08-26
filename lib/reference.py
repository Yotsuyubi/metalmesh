from simulator import Simulation
import meep as mp
import numpy as np
import json
import pickle

C = 299_792_458


def medium_factory(thickness):
    medium = {
        0: [
            mp.Block(
                size=mp.Vector3(mp.inf, mp.inf, thickness),
                center=mp.Vector3(0, 0, 0),
                material=mp.perfect_electric_conductor
            ),
        ],
        90: [
            mp.Block(
                size=mp.Vector3(mp.inf, mp.inf, thickness),
                center=mp.Vector3(0, 0, 0),
                material=mp.perfect_electric_conductor
            )
        ],
    }
    return medium


if __name__ == "__main__":

    with open("sim_param.json", "r") as fp:
        sim_param = json.load(fp)

    a = sim_param["a"]
    resolution = sim_param["resolution"]
    sx = sy = sim_param["sx_sy"]
    sz = sim_param["sz"]
    field_size = (sx, sy, sz)
    f_center = sim_param["f_center"]
    f_width = sim_param["f_width"]
    source_center = sim_param["source_center"]
    pml_width = sim_param["pml_width"]
    monitor_position = sim_param["monitor_position"]
    medium_width = a / resolution

    def simulator_factory(medium, medium_width):
        return Simulation(
            f_center, f_width, source_center,
            field_size, resolution, pml_width,
            medium, medium_width, monitor_position,
            sim_time=80e-12, n_freq=500, a=a
        )

    medium = medium_factory(medium_width / a)
    simulator = simulator_factory(medium, medium_width)

    tran_incidnet, refl_straight = simulator.get_tran_incident()

    with open("refs/tran_incidnet.pickle", mode='wb') as fp:
        pickle.dump(tran_incidnet, fp)
    with open("refs/refl_straight.pickle", mode='wb') as fp:
        pickle.dump(refl_straight, fp)

    for n in range(1, 17):
        medium_width = n * a / resolution
        medium = medium_factory(medium_width / a)
        simulator = simulator_factory(medium, medium_width)
        simulator.refl_straight = refl_straight
        refl_incident = simulator.get_refl_incident()
        with open("refs/refl_incident_{}.pickle".format(n), mode='wb') as fp:
            pickle.dump(refl_incident, fp)
