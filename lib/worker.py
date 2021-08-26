from simulator import Simulation
import meep as mp
import numpy as np
import json
from PIL import Image
import pickle
from tqdm import tqdm
import warnings


C = 299_792_458


def load_mask(path, size):
    im = Image.open(path)
    im = im.resize((size[0], size[1]))
    im = im.convert('L')
    array = np.array(im) / 255
    return array


def arrange_medium(mask, width, height, thickness, a, epsilon):
    medium_list = []
    unit_cell = (width/a) / mask.shape[0]
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[y][x] == 1:
                material = mp.Medium(epsilon_diag=epsilon)
                medium = mp.Block(
                    size=mp.Vector3(unit_cell, unit_cell, thickness/a),
                    center=mp.Vector3(
                        unit_cell*x-(width/a)/2+unit_cell/2,
                        (height/a)/2-unit_cell*y-unit_cell/2, 0
                    ),
                    material=material
                )
                medium_list.append(medium)

    return medium_list


if __name__ == "__main__":

    mp.verbosity(0)

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

    with open("refs/tran_incidnet.pickle", mode='rb') as fp:
        tran_incidnet = pickle.load(fp)
    with open("refs/refl_straight.pickle", mode='rb') as fp:
        refl_straight = pickle.load(fp)

    def simulator_factory(medium, medium_width):
        return Simulation(
            f_center, f_width, source_center,
            field_size, resolution, pml_width,
            medium, medium_width, monitor_position,
            sim_time=250e-12, n_freq=500, a=a
        )

    def medium_factory(thickness, medium0, medium90):
        medium = {
            0: [
                mp.Block(
                    size=mp.Vector3(mp.inf, mp.inf, thickness),
                    center=mp.Vector3(0, 0, 0),
                    material=mp.perfect_electric_conductor
                ),
                *medium0
            ],
            90: [
                mp.Block(
                    size=mp.Vector3(mp.inf, mp.inf, thickness),
                    center=mp.Vector3(0, 0, 0),
                    material=mp.perfect_electric_conductor
                ),
                *medium90
            ],
        }
        return medium

    with open("img/geometry_params.json", "r") as fp:
        geometry_params = json.load(fp)

    offset = 0

    for (index, param) in enumerate(tqdm(geometry_params[offset:]), offset):
        medium_width = param["thickness"] * a / resolution
        mask = load_mask(
            'img/{}'.format(param["filename"]),
            (32, 32)
        )
        medium0 = arrange_medium(
            mask, sx, sy, medium_width,
            a, mp.Vector3(1.69**2, 1.54**2, 1.54**2)
        )
        medium90 = arrange_medium(
            np.rot90(mask), sx, sy, medium_width,
            a, mp.Vector3(1.54**2, 1.69**2, 1.54**2)
        )

        medium = medium_factory(medium_width / a, medium0, medium90)
        simulator = simulator_factory(medium, medium_width)
        with open("refs/refl_incident_{}.pickle".format(param["thickness"]), mode='rb') as fp:
            refl_incident = pickle.load(fp)
        simulator.run(tran_incidnet, refl_incident, refl_straight)

        output_tensor = np.zeros((8, len(simulator.freq)))
        output_tensor[0] = simulator.S11[0].real
        output_tensor[1] = simulator.S11[0].imag
        output_tensor[2] = simulator.S11[1].real
        output_tensor[3] = simulator.S11[1].imag
        output_tensor[4] = simulator.S21[0].real
        output_tensor[5] = simulator.S21[0].imag
        output_tensor[6] = simulator.S21[1].real
        output_tensor[7] = simulator.S21[1].imag
        np.save("spectrums/{}.npy".format(index), output_tensor)
