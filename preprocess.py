import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt

#加载csi文件
def load_csi(path, num_tones = 114, nc = 2, nr = 3) :
    row_obj = pd.read_pickle(path)

    #去除异常值
    row_obj = row_obj[row_obj['csi_len'] != 0]
    row_obj = row_obj[row_obj['num_tones'] == num_tones]
    row_obj = row_obj[row_obj['nc'] == nc]
    row_obj = row_obj[row_obj['nr'] == nr]
    obj = row_obj.reset_index()

    #时间戳处理
    if len(obj) > 0 :
        obj['timestamp'] -= obj['timestamp'][0]
    
    csi_time = np.array(obj['timestamp'])

    #相位矩阵
    csi_series = obj['csi']
    csi_phase = np.empty((0, 2, 3, 114))
    for csi in csi_series :
        csi = csi.transpose()
        angle_csi_temp = np.angle(csi)
        csi_phase = np.concatenate((csi_phase, np.expand_dims(angle_csi_temp, axis=0)))

    return csi_phase, csi_time

#构造相位差矩阵
def diff_phase_matrix(csi_phase, csi_time) :
    phase_temp = np.unwrap(csi_phase, axis=1).transpose() #114*3*2*n
    interval = csi_time[-1] / 1000
    csi_time = csi_time / interval
    xnew = np.array(range(1000))
    csi_phase_temp = np.empty((114, 3, 2, 1000))
    for i in range(114) :
        for j in range(3) :
            for k in range(2) :
                ff = interpolate.interp1d(csi_time, phase_temp[i][j][k])
                csi_phase_temp[i][j][k] = ff(xnew)
    csi_matrix_temp = csi_phase_temp.transpose()
    csi_matrix = np.empty((1000, 114, 3))
    matrix_temp = np.empty((3, 114))
    for i in range(1000) :
        for j in range(3) :
            for k in range(114) :
                diff_temp = csi_matrix_temp[i][0][j][k] - csi_matrix_temp[i][1][j][k]
                if (abs(diff_temp) <= np.pi) :
                    matrix_temp[j][k] = diff_temp
                elif (diff_temp > np.pi) :
                    matrix_temp[j][k] = diff_temp - 2*np.pi
                else:
                    matrix_temp[j][k] = diff_temp + 2*np.pi
        csi_matrix[i] = matrix_temp.transpose()
    return csi_matrix

# 相位差矩阵转图片
def convertToImg(path, csi_matrix) :
    img_matrix = (csi_matrix + np.pi) / (2 * np.pi)
    plt.imsave(path, img_matrix)
    return 

#批量转化
def batch_convert(mode, begin, end) :
    for i in range(begin, end, 2) :
        src_path = '3_12_after/' + mode + '/csi/data_' + str(i) + '.csi'
        dst_path = 'img/' + mode + '/data_' + str(i) + '.jpeg'
        print('processing ' + src_path)
        csi_phase, csi_time = load_csi(src_path)
        csi_matrix = diff_phase_matrix(csi_phase, csi_time)
        #matrix = (csi_matrix + np.pi) / (2 * np.pi)
        convertToImg(dst_path, csi_matrix)
    return 
