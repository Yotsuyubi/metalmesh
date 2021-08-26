import meep as mp
import numpy as np
import sys
import warnings

warnings.simplefilter('ignore')

C = 299_792_458


class Simulation():

    def __init__(
        self,
        # Source frequency settings.
        f_center, f_width, source_center,
        # Simulation field setting in tuple (in um).
        field_size, resolution,
        # PML setting.
        pml_width,
        # Medium setting.
        medium, medium_width,
        # Monitor position.
        monitor_position,
        # Simulation time.
        sim_time=100e3, n_freq=300,
        a=1e-6  # meep unit of length is defined as 1 [um].
    ):

        self.a = a
        self.resolution = resolution
        self.unit_pixel = self.a / self.resolution

        self.sim_time = self.to_meep_time(sim_time)
        self.n_freq = n_freq
        self.f_center = self.to_meep_freq(f_center)
        self.f_width = self.to_meep_freq(f_width)

        self.rr = np.abs(monitor_position) - medium_width/2
        self.rt = np.abs(monitor_position) - medium_width/2
        self.rs = np.abs(source_center) - medium_width/2
        self.d = medium_width

        self.medium = medium
        self.medium_width = self.to_meep_length(medium_width)

        self.sx = self.to_meep_length(field_size[0])
        self.sy = self.to_meep_length(field_size[1])
        self.sz = self.to_meep_length(field_size[2])

        self.field = (self.sx, self.sy, self.sz)

        self.pml_width = pml_width

        self.refl_straight = None

        self.source = [
            mp.Source(
                mp.GaussianSource(
                    self.f_center, fwidth=self.f_width, is_integrated=True
                ),
                component=mp.Ex,
                center=mp.Vector3(0, 0, self.to_meep_length(source_center)),
                size=mp.Vector3(x=self.field[0], y=self.field[0]),
            ),
        ]

        self.simulator = lambda geo: \
            mp.Simulation(
                cell_size=mp.Vector3(*self.field),
                boundary_layers=[
                    mp.PML(self.to_meep_length(self.pml_width), direction=mp.Z)
                ],
                geometry=geo,
                sources=self.source,
                resolution=self.resolution,
                k_point=mp.Vector3()
            )

        self.tran_fr = mp.FluxRegion(
            center=mp.Vector3(0, 0, self.to_meep_length(monitor_position)),
            direction=mp.Y
        )
        self.refl_fr = mp.FluxRegion(
            center=mp.Vector3(0, 0, self.to_meep_length(-monitor_position)),
            direction=mp.Y,
            weight=-1
        )

    def to_meep_length(self, length):
        return length / self.a

    def to_meep_time(self, time):
        return time * C / self.a

    def to_meep_freq(self, freq):
        return freq * self.a / C

    def to_si_length(self, length):
        return length * self.a

    def to_si_freq(self, freq):
        return freq * C / self.a

    def get_tran_incident(self):

        sim = mp.Simulation(
            cell_size=mp.Vector3(*self.field),
            boundary_layers=[
                mp.PML(self.to_meep_length(self.pml_width), direction=mp.Z)
            ],
            geometry=[],
            sources=self.source,
            resolution=self.resolution,
            k_point=mp.Vector3()
        )
        tran_monitor = sim.add_flux(
            self.f_center, self.f_width,
            self.n_freq, self.tran_fr
        )
        refl_monitor = sim.add_flux(
            self.f_center, self.f_width,
            self.n_freq, self.refl_fr
        )
        sim.run(until=self.sim_time)

        self.refl_straight = sim.get_flux_data(refl_monitor)

        return np.array(sim.get_flux_data(tran_monitor).E[:self.n_freq]), sim.get_flux_data(refl_monitor)

    def get_refl_incident(self, debug=True):

        geometry = [
            mp.Block(
                size=mp.Vector3(self.sx, self.sy, self.medium_width),
                center=mp.Vector3(0, 0, 0),
                material=mp.perfect_electric_conductor
            ),
        ]
        sim = mp.Simulation(
            cell_size=mp.Vector3(*self.field),
            boundary_layers=[
                mp.PML(self.to_meep_length(self.pml_width), direction=mp.Z)
            ],
            geometry=geometry,
            sources=self.source,
            resolution=self.resolution,
            k_point=mp.Vector3()
        )
        refl_monitor = sim.add_flux(
            self.f_center, self.f_width,
            self.n_freq, self.refl_fr
        )
        sim.load_minus_flux_data(refl_monitor, self.refl_straight)
        sim.run(until=self.sim_time)

        return np.array(sim.get_flux_data(refl_monitor).E[:self.n_freq])

    def get_signal(self):

        trans = []
        refls = []

        for angle in [0, 90]:

            sim = mp.Simulation(
                cell_size=mp.Vector3(*self.field),
                boundary_layers=[
                    mp.PML(self.to_meep_length(self.pml_width), direction=mp.Z)
                ],
                geometry=self.medium[angle],
                sources=self.source,
                resolution=self.resolution,
                k_point=mp.Vector3()
            )
            tran_monitor = sim.add_flux(
                self.f_center, self.f_width,
                self.n_freq, self.tran_fr
            )
            refl_monitor = sim.add_flux(
                self.f_center, self.f_width,
                self.n_freq, self.refl_fr
            )
            sim.load_minus_flux_data(refl_monitor, self.refl_straight)

            sim.run(until=self.sim_time)

            trans.append(
                sim.get_flux_data(tran_monitor).E[:self.n_freq]
            )
            refls.append(
                sim.get_flux_data(refl_monitor).E[:self.n_freq]
            )

        return \
            np.array(trans), \
            np.array(refls), \
            np.array(mp.get_flux_freqs(tran_monitor))

    def run(self, tran_incidnet, refl_incident, refl_straight):

        warnings.simplefilter('ignore')

        self.refl_straight = refl_straight
        tran_signal, refl_signal, freq = self.get_signal()

        freq = self.to_si_freq(freq)
        vaild_index = np.where((0.5e12 <= freq) & (freq <= 1.0e12))
        self.freq = freq[vaild_index]
        self.S11 = (-1 * refl_signal / refl_incident)[:, vaild_index][:, 0, :]
        self.S21 = (tran_signal / tran_incidnet)[:, vaild_index][:, 0, :]
