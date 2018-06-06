import numpy as np
from skimage.transform import rescale, resize, rotate
import skimage.io as skio
import os
import skimage.color as skcol
import matplotlib.pyplot as plt
import Image
import tqdm

def fliplr(img, points):
    """works with 8 points scaled to 0-1"""
    assert points.shape == (8,2)
    assert points.max() <= 1.0
    imgx = img[:,::-1,:]  # flip image
    switch = np.array([3, 2, 1, 0,  4, 6, 5, 7])
    ptsx = points[switch]
    ptsx = ptsx*[-1,1]+[1, 0]
    return imgx, ptsx


def squarify(H, W, x1, y1, x2, y2):
    a=x2-x1
    b=y2-y1
    if a > b:
        d=(a-b)/2
        y1_ = y1-d
        y2_ = y2+d
        return x1, y1_, x2, y2_
    else:
        d=(b-a)/2
        x1_ = x1-d
        x2_ = x2+d
        return x1_, y1, x2_, y2


def enlargeRect(x1,y1,x2,y2, factor):
    h = y2-y1
    w = x2-x1
    dh = h*(factor-1)/2.0
    dw = w*(factor-1)/2.0
    return x1-dw, y1-dh, x2+dw, y2+dh


def rgb2gray(rgb):
    rgb = rgb.astype(np.float32)/255.
    g = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    return (np.dstack((g,g,g))*255.).astype(np.uint8)


def fixbb(img, pts):
    """img should be uint8"""
    H = img.shape[0]
    W = img.shape[1]
    X_min, X_max, Y_min, Y_max = pts[:,0].min(),\
                                 pts[:,0].max(),\
                                 pts[:,1].min(),\
                                 pts[:,1].max()
    sq = squarify(H, W, X_min, Y_min, X_max, Y_max)
    lM = enlargeRect(sq[0], sq[1],sq[2],sq[3], 1.5)
    pilimg = Image.fromarray(img, mode='RGBA')
    crop = pilimg.crop(tuple(map(lambda x: int(np.round(x)), lM)))
    factor = crop.size[0]/256.0
    crop = crop.resize((256, 256), Image.ANTIALIAS)
    x0, y0 = int(np.floor(lM[0])), int(np.floor(lM[1]))
    newPoints = (pts - np.array([x0, y0]))/factor
    return np.array(crop), newPoints


def get_rand_bg(size, bgpaths):
    """size is tuple"""
    dice = np.random.choice([0, 1, 2], p=[0.1, 0.4, 0.5])
    if dice == 0:
        # random solid color
        bg = np.random.randint(0, 256, (1, 1, 3)).astype(np.uint8)
        bg = resize(bg, (size[0], size[1]), order=0, mode='edge')
        return bg
    elif dice == 1:
        # random noisy texture
        r1 = np.random.choice([2, 3, 4, 8, 16, 32, 128])
        r2 = np.random.choice([2, 3, 4, 8, 16, 32, 128])
        randinterpol = np.random.randint(0, 2)
        bg = np.random.randint(0, 256, (r1, r2, 3)).astype(np.uint8)
        bg = resize(bg, (size[0], size[1]), order=randinterpol, mode='edge')
        bg = rotate(bg, np.random.randint(-90, 91), mode='reflect')
        return bg
    elif dice == 2:
        # random image
        bg = skio.imread(np.random.choice(bgpaths))
        if bg.ndim < 3:
            bg = skcol.gray2rgb(bg)
        bg = rotate(bg, np.random.randint(-5, 5), mode='edge')
        return bg


def background_blend(img):
    backgrounds_folder = '/home/kowalski/projects/synth/datasets/backgrounds/'
    bgs = sorted(os.listdir(backgrounds_folder))
    bgs = [os.path.join(backgrounds_folder, bg) for bg in bgs]

    bg = get_rand_bg(img.shape[:2], bgs)
    if bg.dtype == 'uint8':
        bg = (bg.astype(np.float32) / 255.0)
    if bg.shape[2] == 3:
        bg = np.concatenate((bg, np.ones((256, 256, 1), dtype='float32')), axis=2)

    fg = img.astype(np.float32) / 255.0
    src_rgb = fg[..., :3]
    src_a = fg[..., 3]
    dst_rgb = bg[..., :3]
    dst_a = bg[..., 3]
    out_a = src_a + dst_a * (1.0 - src_a)
    out_rgb = (src_rgb * src_a[..., None]
               + dst_rgb * dst_a[..., None] * (1.0 - src_a[..., None])) / out_a[..., None]
    out = np.zeros_like(bg, dtype=np.float32)
    out[..., :3] = out_rgb
    out[..., 3] = out_a
    return out


def fixlandmarks(landmarks):
    """For landmarks that are 0-1 range and are flipped."""
    pts = landmarks[:, :2]
    pts[:, 1] = 1 - pts[:, 1]
    return pts


def merge_jpg_and_png(bg, fg):
    """bg is jpg and fg is png\noutput is float32"""
    if bg.dtype == 'uint8':
        bg = (bg / 255.0).astype('float32')
    if bg.shape[2] == 3:
        bg = np.concatenate((bg, np.ones((256,256,1), dtype='float32')), axis=2)
    src_rgb = fg[..., :3]
    src_a = fg[..., 3]
    dst_rgb = bg[..., :3]
    dst_a = bg[..., 3]
    out_a = src_a + dst_a*(1.0-src_a)
    out_rgb = (src_rgb*src_a[..., None]
           + dst_rgb*dst_a[..., None]*(1.0-src_a[..., None])) / out_a[..., None]
    out = np.zeros_like(bg)
    out[..., :3] = out_rgb
    out[..., 3] = out_a
    return out


def to_uint8(img):
    return (img[...,:3]*255).astype(np.uint8)


def demo_pred_landmarks(model, datagen, outfolder='', fname='plot.png', num=4):
    X, Y = next(datagen)
    X, Y = X[:num], Y[:num]
    y_pred = model.predict_on_batch(X)
    f, axarr = plt.subplots(1, num)
    for i in range(num):
        img = X[i]
        p = Y[i].reshape((-1, 2)) * 256
        p1 = y_pred[i].reshape((-1, 2)) * 256
        axarr[i].imshow(((img + 1.0) * 127.5).astype(np.uint8))
        axarr[i].scatter(p[:, 0], p[:, 1], c='g')
        axarr[i].scatter(p1[:, 0], p1[:, 1], c='r')
    plt.savefig(os.path.join(outfolder, fname))
    plt.close()


def demo_batch(datagen, num=4):
    X, Y = next(datagen)
    X, Y = X[:num], Y[:num]
    #y_pred = model.predict_on_batch(X)
    f, axarr = plt.subplots(1, num, figsize=(20,5))
    for i in range(num):
        img = X[i]
        p = Y[i].reshape((-1, 2)) * 256
        #p1 = y_pred[i].reshape((-1, 2)) * 256
        axarr[i].imshow(((img + 1.0) * 127.5).astype(np.uint8))
        axarr[i].scatter(p[:, 0], p[:, 1], c='g')
        #axarr[i].scatter(p1[:, 0], p1[:, 1], c='r')
    plt.show()


def demo_prediction(model, dataCSV, outfolder='', fname='plot.png', imSize=256, nCls=2):
    imgPaths = np.array(dataCSV['img'])
    maskPaths = np.array(dataCSV['mask'])
    i = np.random.choice(np.arange(len(imgPaths)))
    img = imgPaths[i]
    msk = maskPaths[i]
    img0 = skio.imread(img)[:,:,:3]
    img = img0.astype(np.float32)/127.5 - 1.0
    msk0 = skio.imread(msk)
    out = model.predict_on_batch(img[None, ...])
    out = out.reshape((imSize, imSize, nCls))
    out = np.argmax(out, axis=-1)
    f, (a1, a2, a3) = plt.subplots(1, 3)
    a1.imshow(img0)
    a2.imshow(msk0)
    a3.imshow(out)
    plt.savefig(os.path.join(outfolder, fname))
    plt.close()


def binary_iou(y1, y2):
    """expected shapes (bs, h * w, 2)"""
    y1 = np.argmax(y1, axis=-1)
    y2 = np.argmax(y2, axis=-1)
    ands = (y1==1) & (y2==1)
    ors = (y1==1) | (y2==1)
    sands = np.sum(ands, axis=1)
    sors = np.sum(ors, axis=1)
    iou = sands.astype(np.float32) / sors.astype(np.float32)
    return np.mean(iou)


def point_score(pts1, pts2, threshold):
    """pts1, pts2 - np.arrays of pixel coordinates nx2 in 0-255 range, threshold in pixel"""
    gotchas = np.sqrt(((pts1-pts2)**2).sum(axis=1)) <= threshold
    return sum(gotchas)*1.0/len(gotchas)


def test_model_real(model, threshold_vals, datagen, steps):
    """threshold_vals is a list of thresholds to test.
    test_images, test_points are the lists of filenames"""
    all_scores = []
    for th in tqdm.tqdm(threshold_vals):
        scores_th = []
        for st in range(steps):
            X, Y = next(datagen)
            pred = model.predict(X)
            for i in range(len(X)):
                ypred = pred[i].reshape(-1, 2)
                pts = Y[i].reshape(-1, 2)
                scores_th.append(point_score(pts*255., ypred*255., th))
        all_scores.append(np.array(scores_th).mean())
    return np.array(all_scores)