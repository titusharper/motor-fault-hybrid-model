import numpy as np

def time_freq_mask(img, T=0.20, F=0.20, num=2):
    h, w, _ = img.shape
    out = img.copy()
    for _ in range(num):
        f = int(F*h*random.random())
        if f > 0:
            f0 = random.randint(0, max(0, h - f))
            out[f0:f0+f, :, :] = 0.0
        t = int(T*w*random.random())
        if t > 0:
            t0 = random.randint(0, max(0, w - t))
            out[:, t0:t0+t, :] = 0.0
    return out

def simple_aug(img):
    if random.random() < 0.30:
        img = time_freq_mask(img, T=0.20, F=0.20, num=2)
    return img

IMAGENET_MEAN = np.array([0.604423, 0.503053, 0.283058], dtype=np.float32)
IMAGENET_STD  = np.array([0.361984, 0.387896, 0.374654], dtype=np.float32)
