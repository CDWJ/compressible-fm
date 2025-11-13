import numpy as np
import imageio.v2 as imageio
import os
import shutil
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from vtk.util import numpy_support
import vtk
from particle_vtk import write_to_vtk
import OpenEXR

def write_particles_vis_valid(pos,pos_valid, outdir, i, particle_num):
    write_to_vtk(
                pos.to_numpy()[:particle_num,:],
                scalar_data={"valid": pos_valid.to_numpy()[:particle_num]},
                vector_data={},
                file_path=os.path.join(outdir,"field_vis_{:03d}".format(i)),
                dim=3
            )
    
def write_particles_vis(pos, outdir, i, particle_num):
    write_to_vtk(
                pos.to_numpy()[:particle_num,:],
                scalar_data={},
                vector_data={},
                file_path=os.path.join(outdir,"field_vis_{:03d}".format(i)),
                dim=3
            )

def write_flip_particles(pos,v,p_type,p_vis_type,p_life, outdir, i, particle_num):
    write_to_vtk(
                pos[:particle_num,:],
                scalar_data={"vis_type":p_vis_type[:particle_num],"type":p_type[:particle_num],"life":p_life[:particle_num]},
                vector_data={"v":v[:particle_num,:]},
                file_path=os.path.join(outdir,"field_vis_{:03d}".format(i)),
                dim=3
            )
    
def write_laden_particles(pos,   outdir, i, particle_num):
    write_to_vtk(
                pos[:particle_num,:],
                scalar_data={},
                vector_data={},
                file_path=os.path.join(outdir,"field_vis_{:03d}".format(i)),
                dim=3
            )
# remove everything in dir
def remove_everything_in(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def write_render_h(h_numpy, outdir, i):
    filename = os.path.join(outdir, "h_field_{:03d}.exr".format(i))
    exr_file = OpenEXR.OutputFile(filename, OpenEXR.Header(h_numpy.shape[1], h_numpy.shape[0]))
    float_array = h_numpy.squeeze().flatten().tobytes()
    exr_file.writePixels({'R': float_array, 'G': float_array, 'B': float_array})
    exr_file.close()

def write_flip_particles_render(pos,p_type, p_life, outdir, i, particle_num):
    # write_to_vtk(
    #             pos[:particle_num,:],
    #             scalar_data={"vis_type":p_vis_type[:particle_num],"type":p_type[:particle_num],"life":p_life[:particle_num]},
    #             vector_data={"v":v[:particle_num,:]},
    #             file_path=os.path.join(outdir,"field_vis_{:03d}".format(i)),
    #             dim=3
    #         )
    pos = pos[:particle_num, :]
    p_type = p_type[:particle_num]
    p_life = p_life[:particle_num]
    idx_to_render = np.where(p_type>=2)[0]
    print(len(idx_to_render))
    print(p_type[idx_to_render].shape)
    write_to_vtk(
            pos[idx_to_render,:],
            scalar_data={"type":p_type[idx_to_render],"life":p_life[idx_to_render]},
            vector_data={},
            file_path=os.path.join(outdir,"field_vis_{:03d}".format(i)),
            dim=3
        )

# for writing images
def to_numpy(x):
    return x.detach().cpu().numpy()
    
def to8b(x):
    return (255*np.clip(x,0,1)).astype(np.uint8)

def comp_vort(vel_img): # compute the curl of velocity
    W, H, _ = vel_img.shape
    dx = 1./H
    u = vel_img[...,0]
    v = vel_img[...,1]
    dvdx = 1/(2*dx) * (v[2:, 1:-1] - v[:-2, 1:-1])
    dudy = 1/(2*dx) * (u[1:-1, 2:] - u[1:-1, :-2])
    vort_img = dvdx - dudy
    return vort_img

def write_image(img_xy, outdir, i):
    img = np.flip(img_xy.transpose([1,0,2]), 0)
    # take the predicted c map
    img8b = to8b(img)
    save_filepath = os.path.join(outdir, '{:04d}.png'.format(i))
    imageio.imwrite(save_filepath, img8b)

def write_levelset_field(img, outdir, particles_pos, vmin=0, vmax=1,  dpi=512//8):
    array = img[:,:,np.newaxis]
    scale_x = array.shape[0]
    scale_y = array.shape[1]
    array = np.transpose(array, (1, 0, 2)) # from X,Y to Y,X
    x_to_y = array.shape[1]/array.shape[0]
    y_size = 7
    fig = plt.figure(num=1, figsize=(x_to_y * y_size + 1, y_size), clear=True)
    ax = fig.add_subplot()
    ax.set_xlim([0, array.shape[1]])
    ax.set_ylim([0, array.shape[0]])
    cmap = 'jet'
    p = ax.imshow(array, alpha = 0.4, cmap=cmap, vmin = vmin, vmax = vmax)
    fig.colorbar(p, fraction=0.046, pad=0.04)
    plt.text(0.87 * scale_x, 0.87 * scale_y, str(0), fontsize = 20, color = "red")
    fig.savefig(os.path.join(outdir, 'levelset_tem.png'), dpi = dpi)
    plt.close()


def write_field(img, outdir, i, particles_pos, cell_type=None, vmin=0, vmax=1, plot_particles=False, dpi=512//8):
    array = img[:,:,np.newaxis]
    scale_x = array.shape[0]
    scale_y = array.shape[1]
    array = np.transpose(array, (1, 0, 2)) # from X,Y to Y,X
    x_to_y = array.shape[1]/array.shape[0]
    y_size = 7
    fig = plt.figure(num=1, figsize=(x_to_y * y_size + 1, y_size), clear=True)
    ax = fig.add_subplot()
    ax.set_xlim([0, array.shape[1]])
    ax.set_ylim([0, array.shape[0]])
    cmap = 'jet'
    p = ax.imshow(array, alpha = 0.4, cmap=cmap, vmin = -3, vmax = 3)
    fig.colorbar(p, fraction=0.046, pad=0.04)
    if plot_particles:
        active_particles_pos = particles_pos
        ax.scatter(active_particles_pos[:, 0], active_particles_pos[:, 1], facecolors='black', s=0.0001)
    plt.text(0.87 * scale_x, 0.87 * scale_y, str(i), fontsize = 20, color = "red")
    fig.savefig(os.path.join(outdir, '{:04d}.png'.format(i)), dpi = dpi)
    plt.close()

def write_h_field(img, outdir, i, particles_pos, cell_type=None, vmin=0, vmax=1, plot_particles=False, dpi=512//8):
    array = img[:,:,np.newaxis]
    scale_x = array.shape[0]
    scale_y = array.shape[1]
    array = np.transpose(array, (1, 0, 2)) # from X,Y to Y,X
    x_to_y = array.shape[1]/array.shape[0]
    y_size = 7
    fig = plt.figure(num=1, figsize=(x_to_y * y_size + 1, y_size), clear=True)
    ax = fig.add_subplot()
    ax.set_xlim([0, array.shape[1]])
    ax.set_ylim([0, array.shape[0]])
    cmap = 'jet'
    p = ax.imshow(array, alpha = 0.4, cmap=cmap, vmin = vmin, vmax = vmax)
    fig.colorbar(p, fraction=0.046, pad=0.04)
    if plot_particles:
        active_particles_pos = particles_pos
        ax.scatter(active_particles_pos[:, 0], active_particles_pos[:, 1], facecolors='black', s=0.0001)
    plt.text(0.87 * scale_x, 0.87 * scale_y, str(i), fontsize = 20, color = "red")
    fig.savefig(os.path.join(outdir, '{:04d}.png'.format(i)), dpi = dpi)
    plt.close()

def write_particles(img, outdir, i, particles_pos, vmin = 0, vmax = 1):
    crop_x = 16
    range_x = [0.5, 1.5]
    crop_y = 5
    range_y = [2.75, 4]
    array = img[:, :, np.newaxis]
    scale_x = array.shape[0]
    scale_y = array.shape[1]
    array = np.transpose(array, (1, 0, 2)) # from X,Y to Y,X
    x_to_y = array.shape[1]/array.shape[0]
    y_size = 7
    fig = plt.figure(num=1, figsize=((x_to_y * y_size + 1) / crop_x * (range_x[1] - range_x[0]),
                                     y_size / crop_y * (range_y[1] - range_y[0])), clear=True)
    fig.subplots_adjust(left=-0.5, right=1, top=1, bottom=0)
    ax = fig.add_subplot()
    # ax.set_xlim([array.shape[1] / crop_x * range_x[0], array.shape[1] / crop_x * range_x[1]])
    # ax.set_ylim([array.shape[0] / crop_y * range_y[0], array.shape[0] / crop_y * range_y[1]])
    ax.set_xlim([0, array.shape[1] / crop_x * (range_x[1] - range_x[0])])
    ax.set_ylim([0, array.shape[0] / crop_y * (range_y[1] - range_y[0])])
    cmap = 'jet'
    p = ax.imshow(array[int(array.shape[0] / crop_y * range_y[0]) : int(array.shape[0] / crop_y * range_y[1]),
                  int(array.shape[1] / crop_x * range_x[0]) : int(array.shape[1] / crop_x * range_x[1]),
                  :], alpha = 0.4, cmap=cmap, vmin = vmin, vmax = vmax)
    fig.colorbar(p, fraction=0.046, pad=0.04)
    particles_pos = particles_pos[(particles_pos[:, 1] > array.shape[0] / crop_y * range_y[0]) &
                                  (particles_pos[:, 1] < array.shape[0] / crop_y * range_y[1]) &
                                  (particles_pos[:, 0] > array.shape[1] / crop_x * range_x[0]) &
                                  (particles_pos[:, 0] < array.shape[1] / crop_x * range_x[1])]

    # contour_width = 2
    # x_array = np.array(particles_pos[:, 0])
    # y_array = np.array(particles_pos[:, 1])
    # centers = np.column_stack((x_array, y_array))
    # circles = patches.Circle(centers, contour_width, edgecolor='black', facecolor='none', lw=contour_width)
    # ax.add_collection(patches.PathCollection(circles))

    # ax.scatter(particles_pos[:, 0],
    #            particles_pos[:, 1],
    #            marker='o', facecolors='black', edgecolors='black', s=0.0001, linewidths=1)

    ax.scatter(particles_pos[:, 0] - array.shape[1] / crop_x * range_x[0],
               particles_pos[:, 1] - array.shape[0] / crop_y * range_y[0],
               marker='o', facecolors='black', edgecolors='black', s=0.0001, linewidths=1)

    plt.text(0.87 * scale_x / crop_x, 0.87 * scale_y / crop_y, str(i), fontsize = 20, color = "red")
    fig.savefig(os.path.join(outdir, '{:04d}.png'.format(i)), dpi = 512)
    plt.close()

def write_w_and_h(w_numpy, h_numpy, outdir, i):
    data = w_numpy.squeeze()
    h_data = h_numpy.squeeze()
    # Create a vtkImageData object and set its properties
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(data.shape)
    imageData.SetOrigin(0, 0, 0)
    imageData.SetSpacing(1, 1, 1)

    # Convert the numpy array to a vtkDataArray and set it as the scalar data
    vtkDataArray = numpy_support.numpy_to_vtk(data.ravel(order = "F"), deep=True)
    vtkDataArray.SetName("vorticity")
    imageData.GetPointData().SetScalars(vtkDataArray)

    # add smoke
    vtkHDataArray = numpy_support.numpy_to_vtk(h_data.ravel(order = "F"), deep=True)
    vtkHDataArray.SetName("h")
    imageData.GetPointData().AddArray(vtkHDataArray)

    # Write the vtkImageData object to a file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(os.path.join(outdir, "field_{:03d}.vti".format(i)))
    writer.SetInputData(imageData)
    writer.Write()

def write_2h(h_numpy, surf_h_numpy, outdir, i):
    data = surf_h_numpy.squeeze()
    h_data = h_numpy.squeeze()
    # Create a vtkImageData object and set its properties
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(data.shape)
    imageData.SetOrigin(0, 0, 0)
    imageData.SetSpacing(1, 1, 1)

    # Convert the numpy array to a vtkDataArray and set it as the scalar data
    vtkDataArray = numpy_support.numpy_to_vtk(data.ravel(order = "F"), deep=True)
    vtkDataArray.SetName("surf_h")
    imageData.GetPointData().SetScalars(vtkDataArray)

    # add smoke
    vtkHDataArray = numpy_support.numpy_to_vtk(h_data.ravel(order = "F"), deep=True)
    vtkHDataArray.SetName("h")
    imageData.GetPointData().AddArray(vtkHDataArray)

    # Write the vtkImageData object to a file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(os.path.join(outdir, "field_{:03d}.vti".format(i)))
    writer.SetInputData(imageData)
    writer.Write()

def write_vol_h(h_numpy,  outdir, i):
    data = h_numpy.squeeze()
    # Create a vtkImageData object and set its properties
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(data.shape)
    imageData.SetOrigin(0, 0, 0)
    imageData.SetSpacing(1, 1, 1)

    # Convert the numpy array to a vtkDataArray and set it as the scalar data
    vtkDataArray = numpy_support.numpy_to_vtk(data.ravel(order = "F"), deep=True)
    vtkDataArray.SetName("h")
    imageData.GetPointData().SetScalars(vtkDataArray)

    # Write the vtkImageData object to a file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(os.path.join(outdir, "field_{:03d}.vti".format(i)))
    writer.SetInputData(imageData)
    writer.Write()

def write_vol_h_vel(h_numpy, u_numpy,  outdir, i):
    data = h_numpy.squeeze()
    print(u_numpy.shape)
    uxdata = u_numpy[...,0].squeeze()
    uydata = u_numpy[...,1].squeeze()
    # Create a vtkImageData object and set its properties
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(data.shape)
    imageData.SetOrigin(0, 0, 0)
    imageData.SetSpacing(1, 1, 1)

    # Convert the numpy array to a vtkDataArray and set it as the scalar data
    vtkDataArray = numpy_support.numpy_to_vtk(data.ravel(order = "F"), deep=True)
    vtkDataArray.SetName("h")
    imageData.GetPointData().SetScalars(vtkDataArray)

    vtkUxDataArray = numpy_support.numpy_to_vtk(uxdata.ravel(order = "F"), deep=True)
    vtkUxDataArray.SetName("ux")
    imageData.GetPointData().AddArray(vtkUxDataArray)

    vtkUyDataArray = numpy_support.numpy_to_vtk(uydata.ravel(order = "F"), deep=True)
    vtkUyDataArray.SetName("uy")
    imageData.GetPointData().AddArray(vtkUyDataArray)

    # Write the vtkImageData object to a file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(os.path.join(outdir, "field_{:03d}.vti".format(i)))
    writer.SetInputData(imageData)
    writer.Write()

def write_vol_h_w(h_numpy, w_numpy,  outdir, i):
    data = h_numpy.squeeze()
    wdata = w_numpy.squeeze()
    # Create a vtkImageData object and set its properties
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(data.shape)
    imageData.SetOrigin(0, 0, 0)
    imageData.SetSpacing(1, 1, 1)

    # Convert the numpy array to a vtkDataArray and set it as the scalar data
    vtkDataArray = numpy_support.numpy_to_vtk(data.ravel(order = "F"), deep=True)
    vtkDataArray.SetName("h")
    imageData.GetPointData().SetScalars(vtkDataArray)

    vtkWDataArray = numpy_support.numpy_to_vtk(wdata.ravel(order = "F"), deep=True)
    vtkWDataArray.SetName("w")
    imageData.GetPointData().AddArray(vtkWDataArray)

    # Write the vtkImageData object to a file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(os.path.join(outdir, "field_{:03d}.vti".format(i)))
    writer.SetInputData(imageData)
    writer.Write()

def write_h( h_numpy, outdir, i):
    data = h_numpy.squeeze()
    # Create a vtkImageData object and set its properties
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(data.shape)
    imageData.SetOrigin(0, 0, 0)
    imageData.SetSpacing(1, 1, 1)

    # Convert the numpy array to a vtkDataArray and set it as the scalar data
    vtkDataArray = numpy_support.numpy_to_vtk(data.ravel(order = "F"), deep=True)
    vtkDataArray.SetName("h")
    imageData.GetPointData().SetScalars(vtkDataArray)


    # Write the vtkImageData object to a file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(os.path.join(outdir, "field_{:03d}.vti".format(i)))
    writer.SetInputData(imageData)
    writer.Write()

def write_mesh_h( v_numpy,f_numpy, outdir, i):
    file_name=os.path.join(outdir, "field_{:03d}.obj".format(i))
    with open(file_name, 'w') as file:
            for v in v_numpy:
                file.write(f"v {v[0]} {v[1]} {v[2]}\n")            
            # Write faces
            for f in f_numpy:
                if(f[0]>=0 and f[1]>=0 and f[2]>=0):
                    file.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")
