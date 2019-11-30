

import os
import random
import shutil

def outputfile(srcfile,outputdir):
    files = os.listdir(srcfile+os.sep+"Annotation")  # 采用listdir来读取所有文件
    random.shuffle(files)

    filesize = len(files)
    oneGroupSize = filesize//5


    i = 0
    for file_name in files:  # 循环读取每个文件名
        i = i + 1
        imageName = file_name[:-3] + "jpg"
        imgUrl = srcfile+os.sep+"Image"+os.sep+imageName
        fileUrl = srcfile+os.sep+"Annotation"+os.sep+file_name

        tagstr1 = ""
        tagstr2 = ""
        # train test val
        if i< (oneGroupSize*3+1):
            # train
            tagstr1 = outputdir + os.sep + "train" + os.sep + "Image" + os.sep
            tagstr2 = outputdir + os.sep + "train" + os.sep + "Annotation" + os.sep

        elif i< (oneGroupSize*4+1):
            # test
            tagstr1 = outputdir + os.sep + "test" + os.sep + "Image" + os.sep
            tagstr2 = outputdir + os.sep + "test" + os.sep + "Annotation" + os.sep

        else:
            # val
            tagstr1 = outputdir + os.sep + "val" + os.sep + "Image" + os.sep
            tagstr2 = outputdir + os.sep + "val" + os.sep + "Annotation" + os.sep

        if not os.path.exists(tagstr1):
            os.makedirs(tagstr1)
        if not os.path.exists(tagstr2):
            os.makedirs(tagstr2)

        # adding exception handling
        try:
            shutil.copy(imgUrl, tagstr1 + os.sep + imageName)
            shutil.copy(fileUrl, tagstr2 + os.sep + file_name)
        except IOError as e:
            print("Unable to copy file. %s" % e)
        except:
            print("Unexpected error:")


if __name__  == "__main__":
    srcdir = "D:\\sysfile\\desktop\\mlbighomework\\"
    outputdir = "D:\\sysfile\\desktop\\mlbighomework\\output\\"
    for i in range(2):
        if i==0:
            outputfile(srcdir+"core_500", outputdir)
        if i==1:
            outputfile(srcdir+"coreless_5000", outputdir)
    print("end")



