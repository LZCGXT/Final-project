import numpy as np
import pandas as pd
from scipy import interpolate, signal
import matplotlib.pyplot as plt
from PIL import Image

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
    csi_amp = np.empty((0, 2, 3, 114))
    for csi in csi_series :
        csi = csi.transpose()
        angle_csi_temp = np.angle(csi)
        #amp_csi_temp = abs(csi)
        amp_csi_temp = 20 * np.log(abs(csi)+1)#db
        csi_phase = np.concatenate((csi_phase, np.expand_dims(angle_csi_temp, axis=0)))
        csi_amp = np.concatenate((csi_amp, np.expand_dims(amp_csi_temp, axis=0)))

    return csi_phase, csi_time, csi_amp

def get_mValue(csi_matrix) : #3*114*1000
    minValue = csi_matrix[0][0][0]
    maxValue = csi_matrix[0][0][0]
    for i in range(3):
        for j in range(114):
            mintemp = min(csi_matrix[i][j])
            maxtemp = max(csi_matrix[i][j])
            if (minValue > mintemp) :
                minValue = mintemp
            if (maxValue < maxtemp) :
                maxValue = maxtemp
    return maxValue, minValue

#构造相位差矩阵和幅度积矩阵(共轭相乘矩阵)
def diff_phase_matrix(csi_phase, csi_time, csi_amp) :
    phase_temp = np.unwrap(csi_phase, axis=1).transpose() #114*3*2*n
    amp_temp = csi_amp.transpose()#114*3*2*n
    interval = csi_time[-1] / 1000
    csi_time = csi_time / interval
    xnew = np.array(range(1000))
    csi_phase_temp = np.empty((114, 3, 2, 1000))
    csi_amp_temp = np.empty((114,3,2,1000))
    for i in range(114) :
        for j in range(3) :
            for k in range(2) :
                ff = interpolate.interp1d(csi_time, phase_temp[i][j][k])
                ff2 = interpolate.interp1d(csi_time, amp_temp[i][j][k])
                csi_phase_temp[i][j][k] = ff(xnew)
                csi_amp_temp[i][j][k] = ff2(xnew)
    csi_matrix_temp = csi_phase_temp.transpose()
    csi_amatrix_temp = csi_amp_temp.transpose()
    csi_matrix = np.empty((1000, 114, 3))
    csi_amatrix = np.empty((1000, 114, 3))
    matrix_temp = np.empty((3, 114))
    amatrix_temp = np.empty((3, 114))
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
                am_temp = csi_amatrix_temp[i][0][j][k] * csi_amatrix_temp[i][1][j][k]
        csi_matrix[i] = matrix_temp.transpose()
        csi_amatrix[i] = am_temp.transpose()
    #csi_amatrix = 20 * np.log(csi_amatrix)#db
    b, a = signal.butter(4, 0.8, 'lowpass',analog=False)
    for i in range(3):
        for j in range(114):
            amplitude_temp = signal.filtfilt(b, a, csi_amatrix.transpose()[i][j],axis=0)
            csi_amatrix.transpose()[i][j] = amplitude_temp
    maxv, minv = get_mValue(csi_amatrix.transpose())
    csi_amatrix = (csi_amatrix - minv) / (maxv - minv)
    return csi_matrix, csi_amatrix

# 相位差矩阵转图片
def convertToImg(path, csi_matrix) :
    img_matrix = (csi_matrix + np.pi) / (2 * np.pi)
    plt.imsave(path, img_matrix)
    return 

#幅度矩阵
def convertToImgC(path, csi_amatrix):
    plt.imsave(path, csi_amatrix)
    return

#C1,C2:phase, C3:amp
def convertToImgA(path, csi_matrix, csi_amatrix) :
    img_matrix = np.empty((3, 114, 1000))
    csi_matrix_temp = (csi_matrix + np.pi) / (2 * np.pi)
    img_matrix[0] = csi_matrix_temp.transpose()[0]
    img_matrix[1] = csi_matrix_temp.transpose()[1]
    img_matrix[2] = csi_amatrix.transpose()[2]
    plt.imsave(path, img_matrix.transpose())
    return

#C1:phase, C2,C3:amp
def convertToImgB(path, csi_matrix, csi_amatrix) :
    img_matrix = np.empty((3, 114, 1000))
    csi_matrix_temp = (csi_matrix + np.pi) / (2 * np.pi)
    img_matrix[0] = csi_matrix_temp.transpose()[0]
    img_matrix[1] = csi_amatrix.transpose()[1]
    img_matrix[2] = csi_amatrix.transpose()[2]
    plt.imsave(path, img_matrix.transpose())
    return

#批量转化
def batch_convert(mode, begin, end) :
    for i in range(begin, end, 2) :
        src_path = '3_12_after/' + mode + '/csi/data_' + str(i) + '.csi'
        dst_path = 'img/' + mode + '/data_' + str(i) + '.jpeg'
        dst_pathA = 'imgA/' + mode + '/data_' + str(i) + '.jpeg'
        dst_pathB = 'imgB/' + mode + '/data_' + str(i) + '.jpeg'
        dst_pathC = 'imgC/' + mode + '/data_' + str(i) + '.jpeg'
        print('processing ' + src_path)
        csi_phase, csi_time, csi_amp = load_csi(src_path)
        csi_matrix, csi_amatrix = diff_phase_matrix(csi_phase, csi_time, csi_amp)
        #matrix = (csi_matrix + np.pi) / (2 * np.pi)
        #convertToImg(dst_path, csi_matrix) #DOWN!
        convertToImgC(dst_pathC, csi_amatrix)#DOWN!
       # convertToImgA(dst_pathA, csi_matrix, csi_amatrix)
        #convertToImgB(dst_pathB, csi_matrix, csi_amatrix)
    return 

#批量压缩114*114
def batch_resize(mode, begin, end) :
    for i in range(begin, end, 2) :
        src_path = 'img/' + mode + '/data_' + str(i) + '.jpeg'
        dst_path = 'gesture_recognition/pic/' + mode + '/data_' + str(i) + '.jpeg'
        print('processing ' + src_path)
        im = Image.open(src_path)
        im2 = im.resize((114, 114))
        im2.save(dst_path)
    return 