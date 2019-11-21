# -*-coding:utf-8-*-\
import os
def loadMyselfAnnotationFile_to_dataset(annotation_file):
    files = os.listdir(annotation_file)  # 采用listdir来读取所有文件

    images = []
    annotations = []
    i=0
    j=0
    for file_name in files:  # 循环读取每个文件名
        i = i + 1
        oneImgInfo = {}
        oneImgInfo["file_name"]=file_name[:-3]+"jpg"
        oneImgInfo["id"] = i
        height = ""
        width = ""
        with open(annotation_file+os.sep+file_name, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                j = j + 1
                oneannotations = {}
                oneannotations["id"] = j
                oneannotations["image_id"] = i
                isCategory = 1
                if "不" in line:
                    isCategory = 2
                oneannotations["category_id"] = isCategory
                oneannotations["segmentation"] = []
                oneannotations["area"] = 0
                oneannotations["iscrowd"] = 0
                linPireList = line.split(" ")
                if len(linPireList) != 6:
                    print('标注数据的行分割后不是6部分')
                x = linPireList[2]
                y = linPireList[3]
                w = linPireList[4]
                h = linPireList[5]
                oneannotations["bbox"] = [int(x), int(y), int(w), int(h)]
                annotations.append(oneannotations)
        images.append(oneImgInfo)
    categories = []
    categoriesdict = {}
    categoriesdict["id"] = 1
    categoriesdict["name"] = "带电芯充电宝"
    categoriesdict["supercategory"] = ""
    categories.append(categoriesdict)
    categoriesdict2 = {}
    categoriesdict2["id"] = 2
    categoriesdict2["name"] = "不带电芯充电宝"
    categoriesdict2["supercategory"] = ""
    categories.append(categoriesdict2)
    dataset = {"info": "用不到","licenses": "用不到"}
    dataset["images"] = images
    dataset["annotations"] = annotations
    dataset["categories"] = categories
    return dataset

if __name__ == "__main__":
    print(loadMyselfAnnotationFile_to_dataset("E:\文件\个人\个人\研究生\课程\机器学习\大作业\core_500\Annotation"))