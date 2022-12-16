import os
import cv2


def AllFiles(DirPath, extension_name='avi'):
    """
    function:
        AllFiles(DirPath, extension_name='avi'):
        讀取目標路徑的所有包含副檔名的檔案

    parameters:
        DirPath: 輸入目標資料夾路徑, str
        extension_name: 目標檔案的附檔名, str. 默認 avi

    return:
        result: 所有符合條件檔案的絕對路徑列表, list
    """
    result = list()
    for root, dirs, files in os.walk(DirPath):
        for f in files:
            if f[-len(extension_name):].lower() == extension_name:
                result.append(os.path.join(root, f))
    return result


def write_video(FrameList, OutputPath, fps=30):
    if len(FrameList[0].shape) == 2:
        raise IOError('輸出影片要為 3 通道影像 (FileIO.py)')

    outputY, outputX, _ = FrameList[0].shape
    video_writer = cv2.VideoWriter(OutputPath, cv2.VideoWriter_fourcc(*'MJPG'), fps, (outputX, outputY))
    for i in FrameList:
        video_writer.write(i)
    video_writer.release()

