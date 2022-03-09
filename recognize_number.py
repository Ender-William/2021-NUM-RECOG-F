import os
import numpy as np
import cv2
from PIL import Image


def start_recognize(model):
    model.summary() #输出模型各层的参数状况
    class_names = ['1','2','3','4','5','6','7','8']  # 这个数组在模型训练的开始会输出
    #要预测的图片保存在这里
    predict_dir = './predict/'
    test = os.listdir(predict_dir)
    if '.DS_Store' in test:
        test.remove('.DS_Store')
    #新建一个列表保存预测图片的地址
    images = []
    #获取每张图片的地址，并保存在列表images中
    for testpath in test:  #循环获取测试路径底下需要测试的图片
        for fn in os.listdir(os.path.join(predict_dir, testpath)):
            if fn.endswith('JPG'):
                fd = os.path.join(predict_dir, testpath, fn)
                #print(fd)
                images.append(fd)
            elif fn.endswith('jpg'):
                fd = os.path.join(predict_dir, testpath, fn)
                #print(fd)
                images.append(fd)
            elif fn.endswith('png'):
                fd = os.path.join(predict_dir, testpath, fn)
                #print(fd)
                images.append(fd)
    result_list = []
    for img in images:
        imgpath = img
        #print(img)
        img_init = cv2.imread(img)
        img_init = cv2.resize(img_init, (224, 224))  # 将图片大小调整到224*224用于模型推理
        cv2.imwrite(img, img_init)
        img = Image.open(img)  # 读取图片
        img = np.asarray(img)  # 将图片转化为numpy的数组
        outputs = model.predict(img.reshape(1, 224, 224, 3))  # 将图片输入模型得到结果
        result_index = int(np.argmax(outputs))
        result = class_names[result_index]
        #result.setText(result)
        #return img,result #返回图片地址和识别出的数字

        imgf = imgpath.split('/')[3]
        imgb = imgf.split('.')[0]
        #print(result)
        result_list.append([imgb,result])

    result_list = sorted(result_list, key=(lambda x:x[0]))

    return result_list #返回二维列表，第一项是照片顺序大小，第二项是识别出后的数字


