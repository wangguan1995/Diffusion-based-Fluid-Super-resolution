import meshio
import matplotlib.pyplot as plt
import numpy as np
dir = "/workspace/wangguan/PaddleScience_Private/Re3900_cylinder/Re3900_0115/Data/slice"
save_dir = "/workspace/wangguan/Diffusion-based-Fluid-Super-resolution/data/re3900/"
def get_low_res(mesh, sorted_indices, var):
    u = mesh.point_data[var][sorted_indices]
    print("u.shape", u.shape)

    u_2d = []
    for i in range(400):
        u_2d.append(u[i * 176 : (i+1) * 176])
    print(np.array(u_2d).shape)
    u_2d = np.array(u_2d).squeeze(-1).T

    downsample = 4
    u_2d_downsampled = u_2d[::downsample, ::downsample]
    u_2d_downsampled = np.repeat(np.repeat(u_2d_downsampled, downsample, axis=0), downsample, axis=1)
    return u_2d, u_2d_downsampled

def plot_data(u_2d, u_2d_downsampled):
    print("u_2d.shape", u_2d.shape)
    print("u_2d_downsampled.shape", u_2d_downsampled.shape)

    import matplotlib.pyplot as plt
    import numpy as np

    plt.imshow(u_2d, cmap='hot', interpolation='nearest')
    plt.savefig('u_2d.png')

    plt.imshow(u_2d_downsampled, cmap='hot', interpolation='nearest')
    plt.savefig('u_2d_downsampled.png')

def generate_data(dir):
    # shape = [10, 3, 176, 400], [time, velocity, x, y]
    velocity_high_res = []
    velocity_low_res = []
    for i in range(200):
        index = 400000 + i * 100
        print("index", index)
        mesh = meshio.read(dir + f"/DNS_slice_{index}.vtu")
        points = mesh.points
        sorted_indices = np.lexsort((points[:,1], points[:,0]))
        u_2d, u_2d_low = get_low_res(mesh, sorted_indices, "u")
        v_2d, v_2d_low = get_low_res(mesh, sorted_indices, "v")
        w_2d, w_2d_low = get_low_res(mesh, sorted_indices, "w")
        velocity_high_res.append(np.stack([u_2d, v_2d, w_2d], axis=0))
        velocity_low_res.append(np.stack([u_2d_low, v_2d_low, w_2d_low], axis=0))
    velocity_high_res = np.array(velocity_high_res)
    velocity_low_res = np.array(velocity_low_res)
    np.save(save_dir + "velocity_high_res.npy", velocity_high_res)
    np.save(save_dir + "velocity_low_res.npy", velocity_low_res)
    print(velocity_low_res.shape)
    print(velocity_high_res.shape)
    plot_data(velocity_high_res[0][1], velocity_low_res[0][1])
        

generate_data(dir)