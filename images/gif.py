import imageio
import numpy as np
import cv2

# fig_dir = "../experiments/re3900_0.12567601/recons__t240_r30_lam0.0/"
fig_dir = "../experiments/jhtdb_re1000/recons__t240_r30_lam0.0/"
img_list = []
for i in range(70):
    # 加载图像
    img1 = cv2.imread(fig_dir + f"sample_batch{i}/input_image.png")
    img2 = cv2.imread(fig_dir + f"sample_batch{i}/NN_predict_image_denosing_step_2.png")
    img3 = cv2.imread(fig_dir + f"sample_batch{i}/reference_image.png")
    # 转换颜色（OpenCV使用BGR，而imageio使用RGB）
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    # 水平叠加图像
    img = np.hstack((img1, img2, img3))
    img_list.append(img)

# imageio.mimsave('re3900.gif', img_list, duration = 0.2) # duration是每帧之间的时间间隔
imageio.mimsave('jhtdb_re1000.gif', img_list, duration = 0.2) # duration是每帧之间的时间间隔
