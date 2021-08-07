import sys
import os
from mpi4py import MPI
from cdql import CDQL
comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

#Use MPI to generate trajectories of trained networks in parallel
if (len(sys.argv) > 1):
    region_num=int(sys.argv[-2])
    target_dist=sys.argv[-1]
    c = CDQL(filename=str(rank) + ".h5", region_num=region_num, target_dist=target_dist)
    print(c.sim_controller.target_dist)
    print(c.sim_controller.region_num)


else:
    c = CDQL(filename=str(rank) + ".h5")

c.model.load_networks(c.folder_name)

if (rank == 0):
    os.system("mkdir " + c.folder_name + "Test")
comm.Barrier()
original_folder_name = c.folder_name

for i in range(0, 100):
    c.folder_name = original_folder_name + "Test/" + str(i*size + rank) + "/"
    os.system("mkdir " + original_folder_name + "Test/" + str(i*size + rank))
    c.test()
