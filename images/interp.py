import numpy as np
# 使用RegularGridInterpolator
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

def F(u, v):
    return u * np.cos(u * v) + v * np.sin(u * v)

fit_points = [np.linspace(0, 3, 8), np.linspace(0, 3, 8)]
values = F(*np.meshgrid(*fit_points, indexing='ij'))

ut, vt = np.meshgrid(np.linspace(0, 3, 80), np.linspace(0, 3, 80), indexing='ij')
true_values = F(ut, vt)
test_points = np.array([ut.ravel(), vt.ravel()]).T

interp = RegularGridInterpolator(fit_points, values)
fig, axes = plt.subplots(2, 4, figsize=(10, 6))
axes = axes.ravel()
fig_index = 0
for method in ['linear', 'nearest', 'slinear', 'cubic', 'quintic', 'pchip']:
    im = interp(test_points, method=method).reshape(80, 80)
    axes[fig_index].imshow(im)
    axes[fig_index].set_title(method)
    axes[fig_index].axis("off")
    fig_index += 1
axes[fig_index].imshow(true_values)
axes[fig_index].set_title("True values")
fig.tight_layout()
fig.show()
plt.savefig('interp.png')
