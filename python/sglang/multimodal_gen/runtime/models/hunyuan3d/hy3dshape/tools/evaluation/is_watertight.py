import igl # pip install libigl
import numpy as np
import trimesh

mesh_file = 'your_mesh.glb'
mesh = trimesh.load(mesh_file, force='mesh', process=False)
F = np.asarray(mesh.faces)
bnd_loops = igl.boundary_loop(F)

if len(bnd_loops) == 0 and igl.is_edge_manifold(F) and igl.is_vertex_manifold(F).all():
    print("判断网格是水密的封闭流形")
else:
    print("非水密网格")
