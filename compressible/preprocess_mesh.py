import sys
import matplotlib.pyplot as plt
import igl
import vtk
import os
from vtk.util import numpy_support
import open3d as o3d

from my_mesh_interpolation import *
from my_lbvh import *
ti.init(arch=ti.cuda, device_memory_GB = 11.0, debug=False)
res_x = 400
res_y = 200
res_z = 200
dx = 1.0 / res_y
center_boundary_mask = ti.field(ti.i32, shape=(res_x, res_y, res_z))
surf_occupancy = ti.field(ti.i32, shape = (res_x, res_y, res_z))
out_occupancy = ti.field(ti.i32, shape = (res_x, res_y, res_z))

# moving solid initialization
carv, carface = igl.read_triangle_mesh(f'./lander.obj')
vn, fn = carv.shape[0], carface.shape[0]
ti_vertices = ti.Vector.field(3, ti.f32, shape = (vn))
ti_faces_0 = ti.Vector.field(3, int, shape = (fn))
changes = ti.field(ti.i32, shape=())

total_length = np.max(np.max(carv, axis=0) - np.min(carv, axis=0))
carv *= (0.21 / total_length)
carv[:, 2] += 0.5
carv[:, 1] += 0.5
carv[:, 0] += 0.65

mesh_to_save = o3d.io.read_triangle_mesh(f'./lander.obj')
mesh_to_save.compute_vertex_normals()
v = np.asarray(mesh_to_save.vertices) * (0.21 / total_length)
# v[:, 2] += 0.5
# v[:, 1] += 0.13
# v[:, 0] += 0.75
v[:, 2] += 0.5
v[:, 1] += 0.5
v[:, 0] += 0.65
mesh_to_save.vertices = o3d.utility.Vector3dVector(v)
o3d.io.write_triangle_mesh("./render_mesh_lander.obj", mesh_to_save)
ti_faces_0.from_numpy(carface)
ti_vertices.from_numpy(carv)

# bvh = LBVH(ti_vertices, ti_faces_0, vn, fn)
# bvh.update_bvh_tree(ti_vertices)
# tribox_ff_voxelize(bvh, surf_occupancy, ti_faces_0, ti_vertices, res_x, res_y, res_z, dx)
# initialize_occupancy(out_occupancy, surf_occupancy, res_x, res_y, res_z)
# changes[None] = 1
# while changes[None]:
#     flood_fill(changes, out_occupancy, surf_occupancy, res_x, res_y, res_z)
# center_boundary_mask.fill(0)
# fill_internal_occupancy(out_occupancy, center_boundary_mask, surf_occupancy)

def init_mesh(res_x, res_y, res_z, scene, dx):
    xs = np.linspace(dx/ 0.5, res_x - dx / 0.5, res_x)
    ys = np.linspace(dx/ 0.5, res_y - dx / 0.5, res_y)
    zs = np.linspace(dx/ 0.5, res_z - dx / 0.5, res_z)
    xv, yv, zv = np.meshgrid(xs, ys, zs, indexing="ij")
    xyz = np.vstack((xv.flatten(), yv.flatten(), zv.flatten())).transpose().astype(np.float32)
    
    occupancy = scene.compute_occupancy(xyz * dx).numpy().reshape(res_x, res_y, res_z)
    # occupancy = scene.contains(xyz/res).reshape(res, res, res)
    # occupancy = trimesh.proximity.signed_distance(scene, xyz/res).reshape(res, res, res)
    return occupancy

mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_to_save)

# Create a scene and add the triangle mesh
scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(mesh)
center_boundary_mask = init_mesh(res_x, res_y, res_z, scene, dx)
# plt.imshow(center_boundary_mask[:, :64, 50])
# plt.show()
np.save('./boundary_mask_plane.npy', center_boundary_mask)
def write_vtk(w_numpy, outdir, i):
    data = w_numpy.squeeze()
    # Create a vtkImageData object and set its properties
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(data.shape)
    imageData.SetOrigin(0, 0, 0)
    imageData.SetSpacing(1, 1, 1)

    # Convert the numpy array to a vtkDataArray and set it as the scalar data
    vtkDataArray = numpy_support.numpy_to_vtk(data.ravel(order = "F"), deep=True)
    vtkDataArray.SetName("Value")
    imageData.GetPointData().SetScalars(vtkDataArray)

    # Write the vtkImageData object to a file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(os.path.join(outdir, "field_{:03d}.vti".format(i)))
    writer.SetInputData(imageData)
    writer.Write()
    


write_vtk(center_boundary_mask, './', 0)