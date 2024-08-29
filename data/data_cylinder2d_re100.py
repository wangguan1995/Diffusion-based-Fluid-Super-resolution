import meshio
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
# z_coordinate = "0.12567601"
# input_dir = f"/workspace/wangguan/PaddleScience_Private/Re3900_cylinder/Re3900_0115/Data/slice_{z_coordinate}"
# save_dir = "/workspace/wangguan/Diffusion-based-Fluid-Super-resolution/data/re3900/"

input_dir = "/workspace/wangguan/Diffusion-based-Fluid-Super-resolution/data/cylinder_2d_re100/cylinder_nektar_wake.mat"
save_dir = "/workspace/wangguan/Diffusion-based-Fluid-Super-resolution/data/cylinder_2d_re100/"
# save_dir = "/workspace/wangguan/Diffusion-based-Fluid-Super-resolution/data/re3900/"

def get_low_res(u_2d):
    u_2d = np.array(u_2d)
    downsample = 4
    u_2d_downsampled = u_2d[::downsample, ::downsample]
    u_2d_downsampled = np.repeat(np.repeat(u_2d_downsampled, downsample, axis=0), downsample, axis=1)
    return u_2d, u_2d_downsampled[:u_2d.shape[0], :u_2d.shape[1]]

def plot_data(u_2d, u_2d_downsampled):
    print("u_2d.shape", u_2d.shape)
    print("u_2d_downsampled.shape", u_2d_downsampled.shape)

    import matplotlib.pyplot as plt
    import numpy as np

    plt.imshow(u_2d, cmap='hot', interpolation='nearest')
    plt.savefig('u_2d.png')

    plt.imshow(u_2d_downsampled, cmap='hot', interpolation='nearest')
    plt.savefig('u_2d_downsampled.png')

def generate_data(input_dir, time=200):
    # shape = [10, 3, 176, 400], [time, velocity, x, y]
    velocity_high_res = []
    velocity_low_res = []
    data = scipy.io.loadmat('./cylinder_2d_re100/cylinder_nektar_wake.mat')
    u = data['U_star'][:,0,:].reshape(50,100,200)
    v = data['U_star'][:,1,:].reshape(50,100,200)
    w = data['p_star'].reshape(50,100,200)

    for i in range(time):
        u_2d, u_2d_low = get_low_res(u[:,:,i])
        v_2d, v_2d_low = get_low_res(v[:,:,i])
        w_2d, w_2d_low = get_low_res(w[:,:,i])
        velocity_high_res.append(np.stack([u_2d, v_2d, w_2d], axis=0))
        velocity_low_res.append(np.stack([u_2d_low, v_2d_low, w_2d_low], axis=0))
    velocity_high_res = np.array(velocity_high_res)
    velocity_low_res = np.array(velocity_low_res)
    np.save(save_dir + f"2d_cylinder_re100_high_res.npy", velocity_high_res)
    np.save(save_dir + f"2d_cylinder_re100_low_res.npy", velocity_low_res)
    print(velocity_low_res.shape)
    print(velocity_high_res.shape)
    plot_data(velocity_high_res[0][0], velocity_low_res[0][0])
        

generate_data(input_dir,time=200)