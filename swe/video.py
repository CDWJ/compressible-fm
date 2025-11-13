import os
import imageio
from PIL import Image
import numpy as np
# 设置生成的视频文件名和路径
filename = 'boat_karman_h2.mp4'
filepath = os.path.join(os.getcwd(), filename)

# 读取所有 PNG 图片
images = []
root="./logs/rotating_triangle/h"
files= os.listdir(root)
files=sorted(files)
#files=files[0:236]
for file_name in files:
    if file_name.endswith('.png'):
        images.append(Image.open(os.path.join(root,file_name)))

# 将图片转换为视频
fps = 10  # 每秒钟30帧
with imageio.get_writer(filepath, fps=fps) as video:
    for i in range(len(images)):
        image=images[i]
        frame = np.array(image.convert('RGB'))#[::-1,:,:]
        video.append_data(frame)