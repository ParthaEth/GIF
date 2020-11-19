import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import model
from PIL import Image

size = 32
img_path = '/is/cluster/scratch/partha/face_gan_data/FFHQ/images1024x1024/46679.png'
img = Image.open(img_path).convert('LA')
img = (np.array(img.resize((size, size))).astype('float32')/255)[:, :, 0]

const_inp = model.ConstantInput(None, size=size)
for i in range(size*2):
    const_inp.const_tensor[0, 0, size//2:-size//2, size//2:-size//2, i] = torch.from_numpy(img)

# in plabe roation, i.e. about z axis
# rot_vec = torch.tensor([0, 0, np.pi/16], dtype=torch.float32)[None, ...]
rot_vec = torch.tensor([np.pi/4, 0, 0], dtype=torch.float32)[None, ...]
rotated = const_inp(rot_vec)

# # plot normal
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(const_inp.const_tensor[0, 0, size//2:-size//2, size//2:-size//2, 3].detach().numpy(), cmap='gray')
# ax1.imshow(img)

# # plot rotated
ax2.imshow(rotated[0, 0, :, :, 3].detach().numpy(), cmap='gray')

# # # plot GRID
# ax3 = fig.add_subplot(111, projection='3d')
# grid = grid.detach().numpy()
# ax3.plot(grid[0, :, :, :, 0], grid[0, :, :, :, 1], grid[0, :, :, :, 2])
plt.show()
