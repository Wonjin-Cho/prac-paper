import pickle

with open('./new/resnet34_refined_gaussian_hardsample_beta0.1_gamma0.5_group1.pickle', 'rb') as f:
    data = pickle.load(f)

# with open('./new/resnet18_labels_hardsample_beta0.1_gamma0.5_group1.pickle', 'rb') as f:
#     labels = pickle.load(f)

print(len(data[0]))
print(type(data[0]))
print(data[0].shape)  # shape: (3, 224, 224)
# If it's a list or dict, inspect the first item
if isinstance(data, dict):
    print(data.keys())

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ls = []
count = np.zeros(1001)
ls = np.zeros(len(data[0]))
# # Example: visualizing the first image
# for i in range(5):
#     for j in range (256):
#         # print(labels[i][j])
#         count[labels[i][j]] += 1
#         if (labels[i][j]==923):
#             ls.append((i,j))

# for i in range(1001):
#     print(f"image label number: {i}")
#     print(count[i])
# data = data[0].cpu().numpy() 
for i in range(len(ls)):
    # img_tensor = data[ls[i][0]][ls[i][1]]  # shape: (3, 224, 224)
    # print(labels[ls[i][0]][ls[i][1]])  # label for the image
    # print(img_tensor.shape)
    img_tensor = data[0][i]  # shape: (3, 224, 224)
    # img_tensor = data[i]
    # print(type(img_tensor))
    # print(img_tensor.shape)
    # print(img_tensor.device)
    img = np.transpose(img_tensor, (1, 2, 0))
    # plt.imshow(img.astype(np.uint8))
    # plt.axis('off')
    # plt.show()

    # To save it as a PNG
    img = img - np.nanmin(img)
    img_max = np.nanmax(img)
    if img_max > 0 and not np.isnan(img_max):
        img = img / (img_max + 1e-8)
    else:
        img = np.zeros_like(img)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img = (img * 255).astype(np.uint8)
    # img_gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    img = Image.fromarray(img)
    # img.save(f'./data/synthetic_img/image_{labels[ls[i][0]][ls[i][1]]}_{i}.png')
    img.save(f'./data/synthetic_img/image_{i}_{i}.png')