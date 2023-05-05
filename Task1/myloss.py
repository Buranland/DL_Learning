import numpy as np

def cross_entropy_loss(y_predict,y_truth):
    delta = 1e-7
    return -(np.mean(y_truth*np.log(y_predict+delta)))
def l1_loss(y_predict,y_truth):
    return np.mean(np.abs(y_predict-y_truth))
def l2_loss(y_predict,y_truth):
    return np.mean(np.square(y_predict-y_truth))
def dice_loss(y_predict,y_truth):
    I = np.sum(y_predict*y_truth)
    U = np.sum(y_predict+y_truth)
    delta = 1e-7
    return 1-(I+delta)/(U-I+delta)
    # return 1-(2*I+delta)/(U+delta)
def focal_loss(y_predict,y_truth,gamma):
    result = -(np.sum(((1-y_predict)**gamma) * np.log(y_predict)))