# 
import numpy as np
# import torch
import imageio.v2 as imageio
import os
import shutil
import glob
import matplotlib.pyplot as plt
from vtk.util import numpy_support
import vtk

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

def to_numpy(x):
    return x.detach().cpu().numpy()
    
def to8b(x):
    return (255*np.clip(x,0,1)).astype(np.uint8)

def write_vtks(w_numpy, smoke_numpy, boundary_mask, outdir, i):
    data = w_numpy.squeeze()
    smoke_data = smoke_numpy.squeeze()
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(data.shape)
    imageData.SetOrigin(0, 0, 0)
    imageData.SetSpacing(1, 1, 1)

    vtkDataArray = numpy_support.numpy_to_vtk(data.ravel(order = "F"), deep=True)
    vtkDataArray.SetName("vorticity")
    imageData.GetPointData().SetScalars(vtkDataArray)

    smokeDataArray = numpy_support.numpy_to_vtk(smoke_data.ravel(order = "F"), deep=True)
    smokeDataArray.SetName("smoke")
    imageData.GetPointData().AddArray(smokeDataArray)

    boundaryDataArray = numpy_support.numpy_to_vtk(boundary_mask.ravel(order = "F"), deep=True)
    boundaryDataArray.SetName("boundary")
    imageData.GetPointData().AddArray(boundaryDataArray)

    # write to file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(os.path.join(outdir, "field_{:03d}.vti".format(i)))
    writer.SetInputData(imageData)
    writer.Write()

def write_field(img, outdir, i, vmin = 0, vmax = 1):
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
    plt.text(0.87 * scale_x, 0.87 * scale_y, str(i), fontsize = 20, color = "red")
    fig.savefig(os.path.join(outdir, '{:03d}.jpg'.format(i)), dpi = 512//8)