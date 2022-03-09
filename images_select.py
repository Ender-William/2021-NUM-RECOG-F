import os
from PIL import Image
import glob
dir1 = './predict/data'  # 图片文件存放地址

def images_select(MAX_WIDTH,MAX_HEIGHT):
    paths = glob.glob(os.path.join(dir1, '*.jpg'))
    # 输出所有文件和文件夹
    for file in paths:
        fp = open(file, 'rb')
        img = Image.open(fp)
        fp.close()
        width = img.size[0]
        height = img.size[1]
        if (width <= MAX_WIDTH) or (height <= MAX_HEIGHT):
            os.remove(file)
