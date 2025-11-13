
#
from hyperparameters import *
from taichi_utils import *
#from mgpcg_solid import *
from init_conditions import *
# from boundary_conditions import *
from io_utils import *
import sys
import shutil
import time
from advect import *
from force import *
#from simple_mesh import *
#from vis_flip import *
from passive_particles import *
from mgpcg import MGPCG_2
from laden_particle import *
import OpenEXR, Imath
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from PIL import Image

ti.init(arch=ti.cuda, device_memory_GB=2.0, debug=False)

h = ti.field(float, shape=(res_x , res_y))
grad_h = ti.field(float, shape=(res_x , res_y))
u_x = ti.field(float, shape=(res_x+1 , res_y))
u_y = ti.field(float, shape=(res_x , res_y+1))
w = ti.field(float, shape=(res_x , res_y))
u = ti.Vector.field(2, float, shape=(res_x , res_y))


def read_exr(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    channels = header['channels'].keys() 
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT) 
    channel_data = {}
    for channel in channels:
        raw_data = exr_file.channel(channel, pt)
        channel_data[channel] = np.frombuffer(raw_data, dtype=np.float32).reshape((height, width))
    if {'R', 'G', 'B'}.issubset(channels):
        rgb_image = np.stack([channel_data['R'], channel_data['G'], channel_data['B']], axis=-1)
        return rgb_image[:,:,0]
    else:
        return channel_data

def write_h_field2(img, data_path,i, vmin=0, vmax=1, dpi=512//8):
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
    fig.savefig(data_path, dpi = dpi)
    plt.close()

def get_rgb(h,vmin,vmax):
    cmap = cm.get_cmap('jet')
    norm = Normalize(vmin=vmin, vmax=vmax)
    normalized_data = norm(h)
    rgb_array = cmap(normalized_data)[:, :, :3]
    return rgb_array 

def save_image(rgb_array,path):
    rgb_array_255 = (rgb_array * 255).astype(np.uint8)
    image = Image.fromarray(rgb_array_255)
    image.save(path)

# main function
def main(from_frame=0, testing=False):
    exp_name  = "Karman2"
    logsdir = os.path.join('.\logs', exp_name)

    ckptsdir = "ckpts"
    ckptsdir  = os.path.join(logsdir, ckptsdir)

    datadir = 'h_exr'
    datadir = os.path.join(logsdir, datadir)

    imagedir = 'img'
    imagedir = os.path.join(logsdir, imagedir)

    os.makedirs(imagedir, exist_ok=True)
    begin_ind = 1
    end_ind = 249
    filename = 'Karman_blended.mp4'
    
    for i in range(begin_ind,end_ind):
        ux_path = os.path.join(ckptsdir , f'vel_x_numpy_{i}.npy')
        uy_path = os.path.join(ckptsdir , f'vel_y_numpy_{i}.npy')
        ux_np = np.load(ux_path)
        uy_np = np.load(uy_path)
        u_x.from_numpy(ux_np)
        u_y.from_numpy(uy_np)
        get_central_vector(u_x, u_y, u)
        curl(u,w,dx)
        file_path = os.path.join(datadir, 'h_field_{:03d}.exr'.format(i))
        img_path = os.path.join(imagedir, 'img_{:03d}.png'.format(i))
        image_data = read_exr(file_path)
        print(image_data.dtype)
        h.from_numpy(image_data)
        calculate_gradient(h,grad_h,dx)
        h_fig = get_rgb(grad_h.to_numpy(),0,0.5)#0.45,0.55)
        #h_fig = get_rgb(h.to_numpy(),0.49,0.51)
        w_fig = get_rgb(w.to_numpy(),-3,3)
        alpha = 0.7
        blended = (w_fig * alpha + (1 - alpha) * h_fig)
        blended = np.swapaxes(blended,0,1)
        save_image(blended,img_path)
        #write_h_field2(w.to_numpy(),img_path,i,0.0, 1.0 )
        print(image_data.shape)
    
    print("finish image.")

    filepath = os.path.join(os.getcwd(), filename)

    # 读取所有 PNG 图片
    images = []
    root=imagedir
    files= os.listdir(root)
    files=sorted(files)
    #files=files[0:236]
    for file_name in files:
        if file_name.endswith('.png'):
            images.append(Image.open(os.path.join(root,file_name)))

    # 将图片转换为视频
    fps = 50  # 每秒钟30帧
    with imageio.get_writer(filepath, fps=fps) as video:
        for i in range(len(images)):
            image=images[i]
            frame = np.array(image.convert('RGB'))#[::-1,:,:]
            video.append_data(frame)


if __name__ == '__main__':
    print("[Main] Begin")
    if len(sys.argv) <= 1:
        main(from_frame=from_frame)
    else:
        main(from_frame=from_frame, testing=testing)
    print("[Main] Complete")