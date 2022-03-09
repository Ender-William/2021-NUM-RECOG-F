import images_cut as ic
import images_select
import recognize_number
import camera

def photo_take_and_recog(model):
    # 要裁剪的照片的保存地址和阈值设置参数
    PATH = './images/IMG_9281.JPG'
    BIN_LOW = 27
    BIN_HIGH = 255

    # 筛选照片的长宽参数
    MAX_WIDTH = 70      #70，80
    MAX_HEIGHT = 80

    #camera.take_pic(PATH,0) #拍照保存路径和相机位
    #按照阈值设置裁剪照片
    ic.images_cut_img_input(PATH,BIN_LOW,BIN_HIGH)
    #筛选出数字牌的区域
    images_select.images_select(MAX_WIDTH,MAX_HEIGHT)

    return recognize_number.start_recognize(model)