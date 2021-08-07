from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
from scipy.stats import gamma
import PIL
import freud
import fresnel
import matplotlib.pyplot as plt
import numpy as np
from mdtraj.reporters import HDF5Reporter
import time


class LJ:
    def __init__(self, filename, region_num=15, target_dist="default_gamma"):
        self.num_particles = 100
        self.dim_length = 30
        self.filename = filename
        self.dimensions = 2
        self.dt = 0.0002
        self.invdt = int(1 / self.dt)
        self.target_dist = target_dist
        self.bin, self.q = self._init_target_distribution(
            dist=self.target_dist)
        self.num_bins = len(self.bin)
        self.system = self._init_system()
        self.integrator = self._init_integrator()
        self.simulation = self._init_simulation()
        self.region_num = region_num  # Along 1 dimension
        self.region_int = np.linspace(0, self.dim_length, self.region_num + 1)
        self.region_temp = np.ones((self.region_num, self.region_num)) * 1.2

    def _init_target_distribution(self, dist="default_gamma"):
        """Initializes the target distribution

        Args:
            dist: The name of the target distribution
        Returns:
            bin: The positions of the endpoints of each bin. Width of each bin
                 is used to calculate probability
            q: The height of each bin
        Raises:
            ValueError: If inputted distribution is not found
        """
        if (dist == "default_gamma"):
            bin = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Regular
            target_data = np.random.gamma(16, 0.25, 10000000)
            q = plt.hist(target_data, bins=(bin + [100]), density=True)
            plt.close()
        else:
            raise ValueError("Dist supplied not defined")
        return bin, q

    def plot_target_distribution(self, dist="default_gamma"):
        """
        Plots target distribution
        Args:
            dist: The name of the target distribution
        Raises:
            ValueError: If inputted distribution is not found
        """
        if (dist == "default_gamma"):
            plt.plot(np.linspace(0, 10, 500), gamma.pdf(
                np.linspace(0, 10, 500), a=16, scale=0.25))
        else:
            raise ValueError("Dist supplied not defined")

    def _init_position(self):
        """Initializes positions on a lattice

        Returns:
            Array of particle positions.
        """
        num_per_dim = round(((self.num_particles)**(1 / self.dimensions))
                            + 0.5)
        lattice_spacing = self.dim_length / num_per_dim
        particle_position = self.num_particles * [0]
        for i in range(self.num_particles):
            x = i % num_per_dim
            y = i // num_per_dim
            x_pos = lattice_spacing * (x + 0.5 * (y % 2))
            y_pos = lattice_spacing * y
            particle_position[i] = Vec3(x_pos, y_pos, 0)

        return particle_position

    def _init_system(self):
        """Initializes an OpenMM system

        Returns:
            Initialized OpenMM System
        """

        a = Quantity((self.dim_length * nanometer,
                     0 * nanometer, 0 * nanometer))
        b = Quantity((0 * nanometer, self.dim_length *
                     nanometer, 0 * nanometer))
        c = Quantity((0 * nanometer, 0 * nanometer,
                     self.dim_length * nanometer))
        system = System()
        system.setDefaultPeriodicBoxVectors(a, b, c)

        sigma = 1 * nanometer
        epsilon = 0.5 * kilojoule_per_mole
        cutoff_type = NonbondedForce.CutoffPeriodic

        lj = CustomNonbondedForce("4*epsilon*(((sigma/r)^12-(sigma/r)^6))")
        lj.addGlobalParameter("sigma", sigma)
        lj.addGlobalParameter("epsilon", epsilon)
        lj.setCutoffDistance(15 * sigma)
        lj.setNonbondedMethod(cutoff_type)

        for particle_index in range(self.num_particles):
            system.addParticle(2 * amu)
            lj.addParticle()

        system.addForce(lj)

        return system

    def _init_integrator(self):
        """Initializes an OpenMM Integrator

        Returns:
            Initialized OpenMM Integrator
        """

        lj_integrator = CustomIntegrator(self.dt)
        lj_integrator.addGlobalVariable("box_length", self.dim_length)
        lj_integrator.addPerDofVariable("D_t", 1.2)
        lj_integrator.addPerDofVariable("dissipation", 0)
        lj_integrator.addPerDofVariable("x_dot", 0)
        lj_integrator.addPerDofVariable("total_force", 0)

        lj_integrator.addComputePerDof("x_dot", "x")
        lj_integrator.addComputePerDof("total_force", "f")
        lj_integrator.addComputePerDof("x", "x + dt*(f) + \
            gaussian * sqrt(2 * D_t * dt)")

        lj_integrator.addComputePerDof("x", "vector(_x(x), _y(x), 0)")
        lj_integrator.addComputePerDof("x_dot", "x - x_dot")
        lj_integrator.addComputePerDof("x_dot", "x_dot + step(x_dot - 0.5*box_length)*(-0.5*box_length)")
        lj_integrator.addComputePerDof("x_dot", "x_dot + step(-(x_dot + 0.5*box_length))*(0.5*box_length)")
        lj_integrator.addComputePerDof(
            "dissipation", "dissipation + (dot(x_dot, total_force)/D_t)")

        lj_integrator.addUpdateContextState()
        return lj_integrator

    def _init_simulation(self):
        """Initializes an OpenMM Simulation

        Returns:
            Initialized OpenMM Simulation
        """
        topology = Topology()
        element = Element.getBySymbol('H')
        chain = topology.addChain()
        for particle in range(self.num_particles):
            residue = topology.addResidue('lj', chain)
            topology.addAtom('lj', element, residue)
        topology.setUnitCellDimensions(
            Quantity(3 * [self.dim_length], nanometer))
        simulation = Simulation(topology, self.system, self.integrator)
        # simulation.context.getPlatform().\
        #     setPropertyDefaultValue("CudaDeviceIndex", "0")
        simulation.context.setPositions(self._init_position())
        simulation.reporters.append(
            HDF5Reporter(self.filename, self.invdt // 100))
        return simulation

    def _get_region_temperature(self, particle_pos):
        """For a given particle position returns temperature of the region that
           particle is in

        Returns:
            Temperature of region particle is in
        """
        x_in = np.sum([self.region_int < particle_pos[0]]) - 1
        y_in = np.sum([self.region_int > particle_pos[1]]) - 1
        return self.region_temp[y_in, x_in]

    def _update_regions(self):
        """Updates temperature of all particles based on the region it is in
        """
        positions = self.simulation.context.getState(
            getPositions=True, enforcePeriodicBox=True).getPositions()
        all_particle_temps = [self._get_region_temperature(x._value)
                              for x in positions]
        temp_vec = [Vec3(particle_i_temp, particle_i_temp, 0)
                    for particle_i_temp in all_particle_temps]

        self.simulation.integrator.setPerDofVariableByName("D_t",
                                                           temp_vec)

    def _color_cluster(self, positions, cl, tag):
        """Renders and saves an image of all clusters of size greater than 2
        Args:
            positions: positions of the particles as a 2D List
            cl: A freud.cluster.Cluster() object of computed clusters
            tag: A string describing the end of the filename of the rendered image
        """

        colors = np.empty((self.num_particles, 3))
        colors[:, :] = fresnel.color.linear([0, 0, 1])
        max = np.max(cl.cluster_idx)
        for i in range(max, 0, -1):
            if (np.sum(cl.cluster_idx == i) > 2):
                break
            colors[cl.cluster_idx == i, :] = fresnel.color.linear([1, 1, 1])
        scene = fresnel.Scene()

        # Spheres for every particle in the system
        geometry = fresnel.geometry.Sphere(scene, N=self.num_particles,
                                           radius=0.5)
        positions = [[pos - (self.dim_length / 2) for pos in row]
                     for row in positions]  # Change left cordinate from 0 to -self.dim_length/2
        geometry.position[:] = positions
        geometry.material = fresnel.material.Material(roughness=0.9)
        geometry.outline_width = 0.05
        # use color instead of material.color
        geometry.material.primitive_color_mix = 1.0
        geometry.color[:] = fresnel.color.linear(colors)
        box = freud.box.Box.square(L=self.dim_length)
        fresnel.geometry.Box(scene, box, box_radius=.1)

        scene.lights = fresnel.light.ring()
        out = fresnel.pathtrace(scene, light_samples=1)
        image = PIL.Image.fromarray(out[:], mode='RGBA')
        filename_clusters = self.filename[:-3] + tag + "_color.png"
        image.save(filename_clusters)

    def update_temperature(self, new_temp, tag=""):
        """Updates self.D_t to be new_temp and saves heatmap of region temperatures
        Args:
            new_temp: 1D (flattened) array of temperatures of regions
            tag: A string describing the end of the filename of the temperature heatmap
        """
        if (not len(new_temp) == (self.region_num ** 2)):
            raise ValueError("Incorrect Action Length")
        self.region_temp = np.array(new_temp).reshape(
            (self.region_num, self.region_num))
        if np.any((self.region_temp <= 0) | (self.region_temp > 2.0)):
            raise ValueError("Unallowed Temperatures Inputted")
        plt.imshow(self.region_temp, cmap="Greys", vmin=0., vmax=2.)
        plt.colorbar()
        filename = self.filename[:-3] + tag + "_temperature.png"
        plt.savefig(filename)
        plt.close()

    def _run_sim(self, time):
        """Runs a simulation for time seconds
        Args:
            time: number of seconds to run simulation
        """
        total_sim_time = int(time * self.invdt)
        self.simulation.step(total_sim_time)

    def _get_KL(self, p):
        """Calculates KL Div from target_distribution to p
        Args:
            p: A normalized distribution of cluster sizes
        Returns:
            KL divergence from target_distribution to p or None if p is None
        Raises:
            ValueError: If q does not have full support over sample space
        """

        if p is None:
            return None
        sum = 0
        ss_len = len(self.q[0])
        for i in range(ss_len):
            p_i = p[0][i] * (p[1][i + 1] - p[1][i])
            q_i = self.q[0][i] * (self.q[1][i + 1] - self.q[1][i])
            try:
                if (p_i == 0):
                    continue
                sum += p_i * np.log(p_i / q_i)
            except:
                raise ValueError("Define q with support over sample space")
        return sum


    def _duplicate_element_by_val(self, count):
        """Duplicates elements by current value. Use to get number of particles per cluster
        E.g. Given an input of [1, 2, 3] it will return [1, 2, 2, 3, 3, 3]
        Args:
            count: A List of all cluster sizes
        Returns:
            A List of the cluster size that each particle belongs to
            or None if the input list is empty (i.e. no clusters present)
        """
        dup_count = []
        for val in count:
            dup_count += [val] * val
        if (len(dup_count) == 0):
            """
            Return None for regions without any particles
            """
            return None
        return dup_count

    def _get_cluster_distribution(self, tag):
        """Gets the distribution of clusters for each region
        Args:
            tag: A string describing the end of the filename
        Returns:
            p: 2D list of normalized distribution of cluster sizes in the entire system
            cs_region: A 3D List of all cluster sizes in each region
        """

        cl = freud.cluster.Cluster()
        box = freud.box.Box.square(L=self.dim_length)
        positions = self.simulation.context.getState(
            getPositions=True, enforcePeriodicBox=True).getPositions()
        positions = [list(x) for x in positions._value]  # Convert to 2D list
        cl.compute((box, positions), neighbors={'r_max': 1.25})  # In nm
        index, counts = np.unique(cl.cluster_idx, return_counts=True)
        cs_region = [[[] for i in range(self.region_num)]
                     for j in range(self.region_num)]
        for p_i in range(self.num_particles):
            particle_pos = positions[p_i]
            x_in = np.sum([self.region_int < particle_pos[0]]) - 1
            y_in = np.sum([self.region_int > particle_pos[1]]) - 1
            current_cluster_index = cl.cluster_idx[p_i]
            # Get all the unique cluster indices in each region
            if current_cluster_index not in cs_region[y_in][x_in]:
                cs_region[y_in][x_in].append(current_cluster_index)

        # Get all the cluster sizes in each region
        cs_region = [[counts[cs_region[i][j]]
                     for j in range(self.region_num)]
                     for i in range(self.region_num)]

        # Get all the particles in a cluster sizes in each region
        cs_region = [[self._duplicate_element_by_val(cs_region[i][j])
                     for j in range(self.region_num)]
                     for i in range(self.region_num)]

        p = [[None if cs_region[i][j] is None else plt.hist(cs_region[i][j],
                                                            bins=self.bin +
                                                            [max(
                                                                max(cs_region[i][j]), self.bin[-1] + 1)],
                                                            density=True)
              for j in range(self.region_num)]
             for i in range(self.region_num)]
        plt.close()
        return p, cs_region

    def _get_cluster_distribution_all(self, tag):
        """Gets the cluster distribution of the entire system (not individual grids)
        Args:
            tag: A string describing the end of the filename
        Returns:
            p: normalized distribution of cluster sizes in the entire system
            counts: A List of all cluster sizes in the entire system
        """
        cl = freud.cluster.Cluster()
        box = freud.box.Box.square(L=self.dim_length)
        positions = self.simulation.context.getState(
            getPositions=True, enforcePeriodicBox=True).getPositions()
        positions = [list(x) for x in positions._value]  # Convert to 2D list
        cl.compute((box, positions), neighbors={'r_max': 1.25})  # In nm
        index, counts = np.unique(cl.cluster_idx, return_counts=True)
        counts = self._duplicate_element_by_val(counts)
        p = plt.hist(counts, bins=self.bin +
                     [max(np.max(counts), self.bin[-1] + 1)], density=True)
        self.plot_target_distribution(dist=self.target_dist)
        filename = self.filename[:-3] + tag + ".png"
        plt.savefig(filename)
        plt.close()
        self._color_cluster(positions, cl, tag)
        return p, counts

    def get_state_reward(self, tag):
        """Returns the current state, reward, and list of cluster sizes of each region
        Args:
            tag: A string describing the end of the filename
        Returns:
            dist: 2D list of normalized distribution of cluster sizes in the entire system
            reward: A 2D list of the KL divergence in each region
            cs_region: A 3D List of all cluster sizes in each region
        """
        p, cs_region = self._get_cluster_distribution(tag)
        reward = []
        dist = []
        for i in range(self.region_num):
            for j in range(self.region_num):
                reward.append(self._get_KL(p[i][j]))
                if (p[i][j] is None):
                    dist.append(None)
                else:
                    curr_dist = p[i][j][0] * np.diff(p[i][j][1])
                    dist.append(curr_dist.tolist())
        return [dist, reward, cs_region]

    def get_state_reward_all(self, tag):
        """Returns the current state, reward, and list of the entire system
        Args:
            tag: A string describing the end of the filename
        Returns:
            dist: list of normalized distribution of cluster sizes in the entire system
            reward: KL divergence of entire system
            cs_region: A List of all cluster sizes in entire system
        """
        p, counts = self._get_cluster_distribution_all(tag)
        reward = self._get_KL(p)
        dist = p[0] * np.diff(p[1])
        state = dist.tolist()
        return [state, reward, counts]

    def run_decorrelation(self, time, tag):
        """Runs a decorrelation step of high temperature to "decorrelate" from some current state
        Args:
            time: time in seconds to run decorrelation
            tag: A string describing the end of the filename
        """
        new_temp = [1.2] * self.region_num**2
        self.update_temperature(new_temp, tag)
        self._update_regions()
        self._run_sim(time)

    def run_step(self, is_detailed=False, tag=""):
        """Runs simulation for one time "step" (i.e. decision) of RL algorithm
        Updates particle activity every 0.25 seconds based on what region particle
        is in. Runs for a total of 0.25 seconds (i.e. 1 decision)
        Args:
            is_detailed: Include information about states/rewards of entire system
            tag: A string describing the end of the filename
        Returns:
            The states, rewards and cluster sizes of the system if is _detailed
            None, None, None if not (is_detailed)
        """
        all_system_rewards = []
        all_system_states = []
        all_system_states_cluster = []
        for i in range(1):
            # Updating once every second
            self._update_regions()
            self._run_sim(0.25)
            if (is_detailed):
                curr_tag = tag + "_" + str(i)
                system_state, system_reward, system_cluster_counts = self.get_state_reward_all(
                    tag)
                all_system_states.append(system_state)
                all_system_rewards.append(system_reward)
                all_system_states_cluster.append(system_cluster_counts)
        if (is_detailed):
            return all_system_states, all_system_rewards, all_system_states_cluster
        else:
            return None, None, None

    def reset_context(self, filename):
        """Resets position to lattice and closes h5 file
        Args:
            filename: file to save new trajectory in
        """

        self.filename = filename
        self.simulation.reporters[0].close()
        self.simulation.reporters[0] = HDF5Reporter(
            self.filename, self.invdt // 100)
        self.simulation.context.setPositions(self._init_position())

    def get_dissipation(self):
        """Gets dissipation of simulation
        Returns:
            Mean total dissipation across all particles
        """
        dissipation = self.simulation.integrator.getPerDofVariableByName(
            "dissipation")
        dissipation = np.array([d_n[0] for d_n in dissipation])
        return np.mean(dissipation)


if __name__ == "__main__":
    lj = LJ("test.h5")
    lj.run_step()
