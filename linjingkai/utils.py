import skimage
import skimage.io
import skimage.transform
import numpy as np
import io
import os


# synset = [l.strip() for l in open('synset.txt').readlines()]


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img


# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1


def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


def test():
    img = skimage.io.imread("./test_data/starry_night.jpg")
    ny = 300
    nx = img.shape[1] * ny / img.shape[0]
    img = skimage.transform.resize(img, (ny, nx))
    skimage.io.imsave("./test_data/test/output.jpg", img)


if __name__ == "__main__":
    test()


def get_cls():
    cls_list = []
    data_path = r'/data0/linjingkai/DataSet/AWA/testclasses.txt'

    cls_file = io.open(data_path, 'r')

    cls_name = ' '
    while cls_name != '':
        # 读取类别的名称并去掉换行符
        cls_name = cls_file.readline().rstrip('\n')
        # 确保没有加入无效的名称
        if len(cls_name) != 0:
            cls_list.append(cls_name)
    print("当前运行类别为：", cls_list)
    print(" ")

    cls_file.close()

    return cls_list

def get_Num(cls):

    path = r'/data0/linjingkai/DataSet/AWA/JPEGImages/' + str(cls)
    pathDir = os.listdir(path)
    count = 0

    for fn in pathDir:  # fn 表示的是文件名
        count = count + 1

    return count

def img_deal(cls , Num):
    batch = []
    path = '/data0/linjingkai/DataSet/AWA/JPEGImages/' + str(cls)
    pathDir = os.listdir(path)

    img = load_image(path + '/' + pathDir[Num])
    batch.append(img)
    Num += 1

    batch_img = np.stack(batch)
    #print(batch_img.shape)
    if(Num%50==0):
        print("已完成：", Num)
        print(" ")

    return batch_img, Num #, label

 #label = []


    # 添加图片路径
   # for img_class in cls:

    #print(path)

    #    for fn in pathDir:  # fn 表示的是文件名
    #        count = count + 1
        #Num = count/1000

        # sample = random.sample(pathDir, 1)
        #count_Num =0

# for name in pathDir:
#     img_path = path + '/' + str(name)
#     #img = skio.imread(img_path)
#     ## 裁剪图像
#     #img_resized = transform.resize(img, (224, 224))
#     #label.append(count)
#    # print(img_resized.shape)
#    # print(name)
#     img = load_image(img_path)

#     batch.append(img)
#     count_Num += 1
#     if count_Num==20:
#         break
#  print("当前类别处理完成：", str(img_class))
# print(count_Num)
# print(" ")
# if count_cls == 0:
#     break

#for u in range(39):
    #    if( Num < img_Num):
    #        img = load_image(path + '/' + pathDir[Num])
    #        batch.append(img)
    #        Num += 1
    #       # print("1234")
    #    else:
    #        break
    #    #count[i] += 1

def read_cls_attr(cls):
    with open('/data0/linjingkai/DataSet/AWA/classes.txt') as file_object:
        index = 0
        for line in file_object:
            #print(str(line))
            if(str(line).strip() == str(cls)):
                Attr_Array = np.loadtxt("/data0/linjingkai/DataSet/AWA/predicate-matrix-binary.txt")
                print("当前类别所在序号：", index)
                return Attr_Array[index]
                break
            index += 1







