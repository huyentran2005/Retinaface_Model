import numpy as np
import torch
import torch.nn as nn


class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** 4.0, 2 ** 6.0, 2 ** 8.0]
        if ratios is None:
            self.ratios = np.array([1, 1, 1])
        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1/2.0) , 2 ** 1.0 ])

    def forward(self, image):
        # image shape: (B, C, H, W)
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        # tính kích thước của feature map ở mỗi pyramid level
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

         # chứa toàn bộ anchors
        all_anchors = np.zeros((0, 4)).astype(np.float32)
        
        # duyệt qua từng pyramid level để tạo anchors
        for idx, p in enumerate(self.pyramid_levels):
            # tạo anchor cơ bản cho level p
            anchors         = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
            # dịch anchor theo từng cell trên feature map
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            # ghép vào danh sách chung
            all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)
        shape: (1, N, 4)
        all_anchors = np.expand_dims(all_anchors, axis=0)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.from_numpy(all_anchors.astype(np.float32)).to(device)

# Tạo anchor cơ sở (reference anchors) cho một pixel điểm của feature map.
def generate_anchors(base_size=16, ratios=None, scales=None):
   
    if ratios is None:
        ratios = np.array([1, 1, 1])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(scales)

    # [x1, y1, x2, y2]
    anchors = np.zeros((num_anchors, 4))

    #  w, h = base_size * scale
    anchors[:, 2:] = base_size * np.tile(scales, (2, 1)).T

    #  (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


#  Dịch từng anchor cơ sở ra mọi vị trí trên feature map.
def shift(shape, stride, anchors):
    # tính vị trí center của từng cell (x,y)
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

     # tạo vector dịch: (dx1, dy1, dx2, dy2)
    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    #A = số lượng anchors cố định (1, A, 4) 
    # K = số cell (tức số vị trí trên feature map)(K, 1, 4) 
    A = anchors.shape[0]
    K = shifts.shape[0]
    # anchor + shift → anchor mới
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    # reshape thành dạng (K*A, 4)
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors    