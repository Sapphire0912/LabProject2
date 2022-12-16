import FileIO
import DCMToAVI
import Preprocessing
import MultiThreshold
import A4CSegmentation
import CalLVEF
import IOU_LVEF

import numpy as np
import time
import cv2
import os


def Process(
        InputDCMDir,
        InputGTDir,
        OutputAVIDir,
        OutputSkeletonizeDir,
        OutputSegmentDir,
        OutputResultDir
):
    """
    function:
        process(
            InputDCMDir,
            InputGTDir,
            OutputAVIDir,
            OutputSkeletonizeDir,
            OutputSegmentDir,
            OutputResultDir
        ):
        控制系統的流程

    parameters:
        InputDCMDir: DCM 檔案的資料夾路徑, str
        InputGTDir: Ground Truth 檔案的資料夾路徑, str
        OutputAVIDir: 輸出存放 avi 檔案的資料夾路徑, str
        OutputSkeletonizeDir: 輸出存放骨架圖檔案的資料夾路徑, str
        OutputSegmentDir: 輸出存放定義完瓣膜和腔室的影片資料夾路徑, str
        OutputResultDir: 輸出存放最後結果的影片資料夾路徑, str
    """

    # 1. DCMToAVI
    ProcessTime = time.time()
    DCMFiles = FileIO.AllFiles(InputDCMDir, 'dcm')
    if len(DCMFiles) == 0:
        raise ValueError(f"{InputDCMDir} 該路徑底下找不到有 DCM 的檔案.")

    FileCount = 0
    for DCMPath in DCMFiles:
        FileCount += 1
        print(f'FileCount: {FileCount}')
        print(f'----- 正在進行 {DCMPath} DCM 轉檔 -----')
        DCMStartTime = time.time()

        ConvertAVI = DCMToAVI.DCMInit(
            FilePath=DCMPath,
            OutputDir=OutputAVIDir
        )

        # 讀取 DCM 裡面的 cine rate & bpm
        CineRate, HeartRate = ConvertAVI.fps, ConvertAVI.bpm
        print(f'CineRate {CineRate}, HeartRate: {HeartRate}')

        if not CineRate:
            print(f'{DCMPath}, DCM 資訊裡沒有 Cine Rate 數值\n')
        if not HeartRate:
            print(f'{DCMPath}, DCM 資訊裡沒有 Heart Rate 數值\n')

        DCMEndTime = time.time()
        print(f'===== DCM 轉檔花費時間: {round(DCMEndTime - DCMStartTime, 2)} 秒 =====\n')

        # 2. AVI 影片的預處理 (ROI & Skeletonize)
        PreStartTime = time.time()
        VideoPath = ConvertAVI.AVIPath
        FileName = str(VideoPath.split('\\')[-1]).replace('DCM', 'avi')
        CaseName = FileName[:-4]

        print(f'----- 正在進行 {VideoPath} AVI 影像的預處理 -----')

        ROI = Preprocessing.ROI(VideoPath)  # return: roi, (ox, oy), radius
        Preprocessing.Skeletonize(VideoPath, OutputSkeletonizeDir)

        PreEndTime = time.time()
        print(f'===== 完成 {VideoPath} Preprocessing 所需時間: {round(PreEndTime - PreStartTime, 2)} 秒 =====\n')
        # 2. AVI 影片的預處理 (ROI & Skeletonize) End.

        # A. MultiThreshold(Unit test Code)
        # test_video = cv2.VideoCapture(VideoPath)
        # test_MT_result = list()
        # while True:
        #     ret, frame = test_video.read()
        #
        #     if not ret:
        #         break
        #
        #     MT = MultiThreshold.MultiThres(frame, ROI[0], 4, 0, 255)
        #     MT.SearchMax()
        #     frame_MT = MT.threshold()
        #     frame_bgr = cv2.cvtColor(frame_MT, cv2.COLOR_GRAY2BGR)
        #     test_MT_result.append(frame_bgr)
        #
        # test_path = "E:\\MyProgramming\\Python\\Project\\implement\\heart recognize2\\A4C TestData" \
        #             "\\MultiThreshold Files\\"
        # test_name = test_path + FileName
        # FileIO.write_video(test_MT_result, test_name, 30)
        # MultiThreshold Unit test End.

        # 3. A4C Segmentation
        SegmentStartTime = time.time()
        SkeletonPath = OutputSkeletonizeDir + FileName + '.png'
        try:
            skeleton = cv2.imread(SkeletonPath)

        except cv2.error:
            print(f'{SkeletonPath}, 沒有找到該檔案的骨架圖')
            continue

        try:
            Segment = A4CSegmentation.Segment(
                VideoPath=VideoPath,
                ROI=ROI,
                OutputSegDir=OutputSegmentDir
            )
            Segment.HandleHeartBound(skeleton)

        except ValueError:
            print(f'{SkeletonPath}, 該骨架圖可能為全黑的')
            continue

        try:
            Segment.Semantic_FindValve(isOutputSegVideo=True)

        except (IndexError, ValueError):
            print(f'{VideoPath}, 統計腔室中心點有問題, 檢查該影片 distance transform 的過程')
            continue

        SegmentEndTime = time.time()
        print(f'===== 完成 {FileName} Segment 所需時間: {round(SegmentEndTime - SegmentStartTime, 2)} 秒 =====\n')
        # 3. A4C Segmentation End.

        # 4. 計算 LV Bound
        print(f'----- 正在計算 {FileName} LVEF -----')
        LV_pt = CalLVEF.TimeMedian(Segment.LeftPivotList)
        RV_pt = CalLVEF.TimeMedian(Segment.RightPivotList)
        LV_cn = CalLVEF.TimeMedian(Segment.HistoryCenters["LV"])
        lastL, lastR = CalLVEF.LVBound(VideoPath, ROI[0], LV_cn, LV_pt, RV_pt)

        # 5. 檢查結果, 並將結果輸出到指定資料夾
        cap = cv2.VideoCapture(VideoPath)
        frame_count = 0
        all_frames = list()
        result_dict = dict()

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            FinLR = list()
            for i in range(0, 20):
                lastX, lastY = int(lastR[frame_count][i][0]), int(lastR[frame_count][i][1])
                cv2.circle(frame, (lastX, lastY), 3, (240, 153, 216), -1)
                FinLR.append((lastX, lastY))

            for i in range(19, -1, -1):
                lastX, lastY = int(lastL[frame_count][i][0]), int(lastL[frame_count][i][1])
                cv2.circle(frame, (lastX, lastY), 3, (240, 153, 216), -1)
                FinLR.append((lastX, lastY))

            result_area = np.zeros([600, 800], dtype=np.uint8)
            for i in range(0, len(FinLR), 1):
                if i + 1 == len(FinLR):
                    cv2.line(result_area, (FinLR[i][0], FinLR[i][1]),
                             (FinLR[0][0], FinLR[0][1]), (255, 255, 255), 1)

                else:
                    cv2.line(result_area, (FinLR[i][0], FinLR[i][1]),
                             (FinLR[i + 1][0], FinLR[i + 1][1]), (255, 255, 255), 1)

            cnt, hierarchy = cv2.findContours(result_area, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cnt, key=cv2.contourArea)
            cv2.drawContours(result_area, [c], -1, (255, 255, 255), -1)
            result_dict[frame_count + 1] = result_area  # 利用字典儲存每幀的結果, 以提供給計算 LVEF 的輸入

            # unit test
            area_bgr = cv2.cvtColor(result_area, cv2.COLOR_GRAY2BGR)
            frame = cv2.addWeighted(frame, 1, area_bgr, 0.4, 0)
            frame_count += 1
            cv2.putText(frame, "frame count:%d" % frame_count, (70, 60),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
            all_frames.append(frame)

        OutputResultPath = OutputResultDir + FileName
        FileIO.write_video(all_frames, OutputResultPath)

        # 6. 計算 LVEF 的結果
        # create UnitCM List
        UnitList = [
            17, 17, 17, 16, 15, 15, 15,
            13, 15, 15, 15, 15, 15, 15,
            15, 15, 17, 15, 17, 13, 13
        ]

        # call IOU_LVEF
        # 1. compare two LVEF result
        GTDir = InputGTDir + CaseName + '\\'
        lvef = IOU_LVEF.LVEFInfo(
            UnitCM=UnitList[FileCount - 1],
            Radius=ROI[2]
        )
        lvef.FindCycle(result_dict, CineRate, HeartRate)
        print("ESV Frame:", lvef.ESVFrame, "EDV Frame:", lvef.EDVFrame)
        # draw Dice value

        try:
            EDV, ESV, LVEF, Degree = lvef.calLVEF(mode='self')  # self lvef
            GT_EDV, GT_ESV, GT_LVEF, GT_Degree = lvef.calLVEF(mode='GT', GTDir=GTDir)
            print(f'Self -> EDV: {EDV}cm\u00b3, ESV: {ESV}cm\u00b3, LVEF: {LVEF}%, Degree: {Degree}')
            print(f'GT -> EDV: {GT_EDV}cm\u00b3, ESV: {GT_ESV}cm\u00b3, LVEF: {GT_LVEF}%, Degree: {GT_Degree}')

        except IndexError:
            print("!!!!! 需要計算 LVEF !!!!!")
            EDVTime, ESVTime = lvef.EDVFrame, lvef.ESVFrame
            EDVFrame = [result_dict[i] for i in EDVTime]
            ESVFrame = [result_dict[i] for i in ESVTime]

            GTFiles = FileIO.AllFiles(GTDir, 'png')
            GTDict = dict()
            for files in GTFiles:
                fc = int(files.split('_')[-1].split('.')[0])  # fc -> frame_count
                GTDict[fc] = files

            EDVArea, ESVArea = list(), list()
            for i, _mask in enumerate(EDVFrame):
                cnt, _ = cv2.findContours(_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                _area = cv2.contourArea(cnt[0])
                EDVArea.append(_area)

            for i, _mask in enumerate(ESVFrame):
                cnt, _ = cv2.findContours(_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                _area = cv2.contourArea(cnt[0])
                ESVArea.append(_area)

            Text = ["Hyper Dynamic", "Normal", "Mild", "Moderate", "Severe"]
            Degree = np.array([70, 49, 39, 29, 0])
            EDV, ESV = max(EDVArea), min(ESVArea)
            EDV = round(EDV / (UnitList[FileCount - 1] ** 3), 1)
            ESV = round(ESV / (UnitList[FileCount - 1] ** 3), 1)

            LVEF = round(100 * (EDV - ESV) / EDV, 1)
            degree = Text[len(Degree[Degree > LVEF])]

            # GT LVEF
            GT_EDVList, GT_ESVList = list(), list()
            for i in EDVTime:
                _mask = cv2.imread(GTDict[i], 0)
                cnt, _ = cv2.findContours(_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                _area = cv2.contourArea(cnt[0])
                GT_EDVList.append(_area)

            for i in ESVTime:
                _mask = cv2.imread(GTDict[i], 0)
                cnt, _ = cv2.findContours(_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                _area = cv2.contourArea(cnt[0])
                GT_ESVList.append(_area)

            GT_EDV, GT_ESV = max(GT_EDVList), min(GT_ESVList)
            GT_EDV = round(GT_EDV / (UnitList[FileCount - 1] ** 3), 1)
            GT_ESV = round(GT_ESV / (UnitList[FileCount - 1] ** 3), 1)
            GT_LVEF = round(100 * (GT_EDV - GT_ESV) / GT_EDV, 1)
            GT_degree = Text[len(Degree[Degree > GT_LVEF])]

            print(f'Self -> EDV: {EDV}cm\u00b3, ESV: {ESV}cm\u00b3, LVEF: {LVEF}%, Degree: {degree}')
            print(f'GT -> EDV: {GT_EDV}cm\u00b3, ESV: {GT_ESV}cm\u00b3, LVEF: {GT_LVEF}%, Degree: {GT_degree}')
        print("===== End. =====\n")

        # write img
        # target = cv2.VideoCapture(VideoPath)
        # _counts = 0
        # _writeDir = "E:\\MyProgramming\\Python\\Project\\implement\\heart recognize2\\A4C TestData\\TargetDir\\"
        # while True:
        #     ret, frame = target.read()
        #
        #     if not ret:
        #         break
        #
        #     _counts += 1
        #     _casePath = os.path.join(_writeDir, CaseName)
        #     if not os.path.exists(_casePath):
        #         os.makedirs(_casePath)
        #
        #     _framestr = str(_counts) if _counts > 9 else "0%s" % _counts
        #     _filename = FileName.split('.')[0]
        #     _concat = _filename + '_' + _framestr + '.png'
        #     cv2.imwrite(_casePath + "\\" + _concat, result_dict[_counts])
