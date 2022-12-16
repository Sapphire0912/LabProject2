from VideoROI import *
from MultiThreshold import *
import cv2
import numpy as np


def merge_video(F1, F2, OutputPath):
    F1Y, F1X, _ = F1[0].shape
    F2Y, F2X, _ = F2[0].shape

    video_writer = cv2.VideoWriter(OutputPath, cv2.VideoWriter_fourcc(*'MJPG'), 20, (F1X + F2X, F1Y))
    for i in range(len(F1)):
        merge = np.hstack([F1[i], F2[i]])
        video_writer.write(merge)
    video_writer.release()


def AllFiles(InputDirPath):
    FilesList = list()
    for root, dirs, files in os.walk(InputDirPath):
        for f in files:
            if f[-3:].lower() == 'avi':
                FilesList.append(os.path.join(root, f))

    return FilesList


def write_video(Frame_List, OutputPath):
    outputY, outputX, _ = Frame_List[0].shape
    video_writer = cv2.VideoWriter(OutputPath, cv2.VideoWriter_fourcc(*'MJPG'), 20, (outputX, outputY))
    for i in Frame_List:
        video_writer.write(i)
    video_writer.release()


# Home
A4CDirPath = 'E:\\MyProgramming\\Python\\Project\\implement\\heart recognize\\Heart Bound\\Test A4C Video\\'
PLADirPath = 'E:\\MyProgramming\\Python\\Project\\implement\\heart recognize\\Heart Bound\\Test PLA Video\\'
ALADirPath = 'E:\\MyProgramming\\Python\\Project\\implement\\heart recognize\\Heart Bound\\Test ALA Video\\'

A4COutputDirPath = 'E:\\MyProgramming\\Python\\Project\\implement\\heart recognize\\MultiThreshold_ALL\\test MT A4C\\'
PLAOutputDirPath = 'E:\\MyProgramming\\Python\\Project\\implement\\heart recognize\\MultiThreshold_ALL\\test MT PLA\\'
ALAOutputDirPath = 'E:\\MyProgramming\\Python\\Project\\implement\\heart recognize\\MultiThreshold_ALL\\test MT ALA\\'

VideoPath = AllFiles(A4CDirPath)

for Path in VideoPath:
    FileName = str(Path.split('\\')[-1])
    print(f'FileName: {FileName}')

    Init = VideoInit(Path)
    MaskROI = Init.roi
    FrameList = list()
    FrameOri = list()
    frame_count = 0

    while 1:

        ret, frame = Init.video.read()

        if not ret:
            break

        frame[MaskROI != 255] = [0, 0, 0]
        frame_display = frame.copy()
        frame_count += 1

        # --- MultiThreshold v3
        Multi = MultiThres(frame, MaskROI, 4, 0, 255)
        Multi.SearchMax()
        # print(f'Value Avg: {Multi.ValueList}')
        MultiFrame = Multi.threshold()
        # --- MultiThreshold v3 End.

        MultiFrame = cv2.cvtColor(MultiFrame, cv2.COLOR_GRAY2BGR)
        cv2.putText(MultiFrame, f'frame_count: {frame_count}', (40, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
        cv2.putText(frame_display, f'frame_count: {frame_count}', (40, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
        # cv2.rectangle(frame_display, (350, 80), (450, 150), (0, 255, 0), 2)

        # FrameList.append(MultiFrame)
        # FrameOri.append(frame_display)
        # cv2.imshow('frame', frame_display)
        # cv2.imshow('MT', MultiFrame)
        #
        # if cv2.waitKey(0) == ord('n'):
        #     continue
        # elif cv2.waitKey(0) == ord('q'):
        #     break
        # pass

    # write_path = A4COutputDirPath + ' MT2 ' + FileName
    # write_path = PLAOutputDirPath + ' MT2 ' + FileName
    # write_path = ALAOutputDirPath + ' MT2 ' + FileName

    # merge_video(FrameOri, FrameList, write_path)
