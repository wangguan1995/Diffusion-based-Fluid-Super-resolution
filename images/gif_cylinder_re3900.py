import imageio
import numpy as np
import cv2
import os

input_fig_dir = "./cylinder_re3900_input"
reference_fig_dir = "./cylinder_re3900_reference"
predict_fig_dir = "../experiments/cylinder_re3900_UNet_1/recons__t240_r30_lam0.0/"
img_list = []
os.makedirs('./cylinder_re3900_UNet_output', exist_ok=True)  
for i in range(180, 200):
    # 加载图像
    img1 = cv2.imread(input_fig_dir         + f"/input_image_t_{i}.png")
    img3 = cv2.imread(reference_fig_dir     + f"/reference_image_t_{i}.png")
    img2 = cv2.imread(predict_fig_dir       + f"/predict_t_{i-180}.png")
    # 转换颜色（OpenCV使用BGR，而imageio使用RGB）
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    # 水平叠加图像
    img = np.hstack((img1, img2, img3))
    img_list.append(img)
    imageio.imwrite(f'./cylinder_re3900_UNet_output/t_{i}.png', img)
imageio.mimsave('re3900.gif', img_list, duration = 0.2) # duration是每帧之间的时间间隔
