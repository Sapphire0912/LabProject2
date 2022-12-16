import FileIO
import MultiThreshold as Mt
import MuscleSampling as Vp

import numpy as np
import cv2
import os


def A4CModel(width, height):  # 剩下給予模型中每一段位置的名稱 (semantic name)
    """
        A4CModel(width, height):
            Handel Standard Model (Muscle + Chamber)

        parameters:
            width: 調整模型比例的寬度
            height: 調整模型比例的長度

        return:
            A4CModelInfo: dict, contains Split_9_Model, UpperModel, LowerModel and ChamberModel
            information for centers, contours, position and src.
    """

    A4CModelInfo = {
        "Split_9_Model":
            {"Centers": np.array([]), "Contours": list(), "Position": list(), "src": np.array([])},
        "UpperModel":
            {"Centers": np.array([]), "Contours": list(), "Position": list(), "src": np.array([])},
        "LowerModel":
            {"Centers": np.array([]), "Contours": list(), "Position": list(), "src": np.array([])},
        "LVModel":
            {"Centers": np.array([]), "Contours": list(), "Position": list(), "src": np.array([])},
        "ChamberModel":
            {"Centers": np.array([]), "Contours": list(), "Position": list(), "src": np.array([])}
    }

    # --- A4C 肌肉模型部分
    ModelPath = "E:\\MyProgramming\\Python\\Project\\implement\\heart recognize\\System2\\model" \
                "\\0002_Apical four chamber(no valve)_Split9.png"
    A4C = cv2.imread(ModelPath)

    # -- 處理 A4C 模型, 切成 9 等分(利用小畫家切)
    ReA4C = cv2.resize(A4C, (width, height), cv2.INTER_AREA)
    gray = cv2.cvtColor(ReA4C, cv2.COLOR_BGR2GRAY)

    _, thres = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    CntModel, _ = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ModelCenter = list()
    for index, cnt in enumerate(CntModel):
        M = cv2.moments(cnt)
        Cx, Cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        ModelCenter.append([Cx, Cy])
    ModelCenter = np.asarray(ModelCenter)

    # 按照左 -> 右, 上 -> 下 排列 (目前按照上 -> 下排列即可定義)
    # index 即是定義好的位置標記
    sortY = np.sort(ModelCenter[:, 1])
    SortCenter = np.zeros(ModelCenter.shape, np.int)
    SortCntModel = list()  # 將原本的 Model Contours 按照 SortCenter 順序排好

    for index, val in enumerate(sortY):
        pos = np.argwhere(ModelCenter[:, 1] == val)[0][0]
        SortCenter[index] = ModelCenter[pos]
        SortCntModel.append(CntModel[pos])
    # End.

    # - 將模型區分為上下兩等分
    UpperCntModel = SortCntModel[:5]
    maskUpperModel = np.zeros(gray.shape, np.uint8)

    for CntUpper in UpperCntModel:
        cv2.drawContours(maskUpperModel, [CntUpper], -1, (255, 255, 255), -1)

    maskLowerModel = cv2.bitwise_xor(maskUpperModel, thres)
    # - 將模型區分為上下兩等分 End.

    # - Only LV
    LVCntModel = [SortCntModel[1], SortCntModel[3], SortCntModel[4]]
    maskLVModel = np.zeros(gray.shape, np.uint8)

    for CntLV in LVCntModel:
        cv2.drawContours(maskLVModel, [CntLV], -1, (255, 255, 255), -1)
    # -- 處理 A4C 模型, 切成 9 等分
    # --- A4C 肌肉模型部分 End.

    # --- A4C 模型腔室部分
    ModelPath2 = "E:\\MyProgramming\\Python\\Project\\implement\\heart recognize\\System3\\model\\" \
                 "0002_Apical four chamber (chamber).png"
    A4CChamber = cv2.imread(ModelPath2)

    # -- 處理 A4C 模型, 切成 4 等分(利用小畫家切)
    ReA4CChamber = cv2.resize(A4CChamber, (width, height), cv2.INTER_AREA)
    grayChamber = cv2.cvtColor(ReA4CChamber, cv2.COLOR_BGR2GRAY)

    _, thresChamber = cv2.threshold(grayChamber, 180, 255, cv2.THRESH_BINARY)
    regCnt, _ = cv2.findContours(thresChamber, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(grayChamber.shape, np.uint8)
    cv2.drawContours(mask, regCnt, -1, (255, 255, 255), -1)
    thresChamber = cv2.bitwise_not(thresChamber, mask=mask)
    CntChamber, _ = cv2.findContours(thresChamber, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ModelCbCenter = list()
    for index, cnt in enumerate(CntChamber):
        M = cv2.moments(cnt)
        Cx, Cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        ModelCbCenter.append([Cx, Cy])
    ModelCbCenter = np.asarray(ModelCbCenter)

    # 按照左 -> 右, 上 -> 下 排列
    sortY = np.sort(ModelCbCenter[:, 1])
    SortCbCenter = np.zeros(ModelCbCenter.shape, np.int)
    SortCntCbModel = list()

    for index, val in enumerate(sortY):
        pos = np.argwhere(ModelCbCenter[:, 1] == val)[0][0]
        SortCbCenter[index] = ModelCbCenter[pos]
        SortCntCbModel.append(CntChamber[pos])
    # End.
    # -- 處理 A4C 模型, 切成 4 等分 End.
    # --- A4C 模型腔室部分 End.

    # 給值
    A4CModelInfo["Split_9_Model"]["Centers"] = SortCenter
    A4CModelInfo["Split_9_Model"]["Contours"] = SortCntModel
    A4CModelInfo["Split_9_Model"]["Position"] = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 到時給予每一段名稱
    A4CModelInfo["Split_9_Model"]["src"] = gray

    A4CModelInfo["UpperModel"]["Centers"] = SortCenter[:5]
    A4CModelInfo["UpperModel"]["Contours"] = UpperCntModel
    A4CModelInfo["UpperModel"]["Position"] = [0, 1, 2, 3, 4]  # 到時給予每一段名稱
    A4CModelInfo["UpperModel"]["src"] = maskUpperModel

    A4CModelInfo["LowerModel"]["Centers"] = SortCenter[5:]
    A4CModelInfo["LowerModel"]["Contours"] = SortCntModel[5:]
    A4CModelInfo["LowerModel"]["Position"] = [5, 6, 7, 8]  # 到時給予每一段名稱
    A4CModelInfo["LowerModel"]["src"] = maskLowerModel

    A4CModelInfo["LVModel"]["Centers"] = [SortCenter[1], SortCenter[3], SortCenter[4]]
    A4CModelInfo["LVModel"]["Contours"] = LVCntModel
    A4CModelInfo["LVModel"]["Position"] = [1, 3, 4]  # 到時給予每一段名稱
    A4CModelInfo["LVModel"]["src"] = maskLVModel

    A4CModelInfo["ChamberModel"]["Centers"] = SortCbCenter
    A4CModelInfo["ChamberModel"]["Contours"] = SortCntCbModel
    A4CModelInfo["ChamberModel"]["Position"] = [0, 1, 2, 3]  # 到時給予每一段名稱
    A4CModelInfo["ChamberModel"]["src"] = thresChamber

    return A4CModelInfo


def ModelMatching(src, target, vertical, horizontal, theta):
    """
        ModelMatching(src, target, vertical, horizontal, theta):
            Matching Standard Model, returning best fit information.

        parameters:
            src: Multi-Threshold image (gray shape)
            target: Standard Model (gray shape)
            vertical: range(min, max, step), iterator
            horizontal: range(min, max, step), iterator
            theta: range(min, max, step), iterator

        return:
            modelBest: Best fit matrix.
            BestFitting: src and target best fit area.
            (BestVertical, BestHorizontal, BestAngle): Best each parameter.
    """

    BestVertical, BestHorizontal, BestAngle = 0, 0, 0
    MaxFittingArea = 0
    BestModel = None

    BestFitting = np.zeros(src.shape, np.uint8)  # gray shape
    Height, Width = src.shape

    for vert in vertical:
        for hori in horizontal:
            transMat = np.array([[1, 0, hori], [0, 1, vert]], np.float32)
            affineMat = cv2.warpAffine(target, transMat, (Width, Height))

            for angle in theta:
                rotateMat = cv2.getRotationMatrix2D((Width / 2, Height / 2), angle, 1)
                rotateModel = cv2.warpAffine(affineMat, rotateMat, (Width, Height))

                fitting = src.copy()
                fitting[rotateModel != 255] = 0
                fittingArea = np.sum(fitting)

                if fittingArea > MaxFittingArea:
                    MaxFittingArea = fittingArea
                    BestVertical, BestHorizontal, BestAngle = vert, hori, angle

                    BestFitting = fitting
                    BestModel = rotateModel

    return BestModel, BestFitting, (BestVertical, BestHorizontal, BestAngle)


class MatchModel(object):
    def __init__(self, Path, ROI, OutputMatchingDir):
        # video information
        self.video = cv2.VideoCapture(Path)
        self.Path = Path
        self.Height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.Width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.OutputMatchingDir = OutputMatchingDir

        self.roi = ROI[0]
        self.ox, self.oy = ROI[1]
        self.Info = None

        # mitral valve position
        self.LeftMValve = list()
        self.RightMValve = list()

        # 紀錄 Muscle Semantic Sample Points
        self.MuscleSemantic = {
            "Apical Septal": list(),
            "Septal": list(),
            "Basal Septal": list(),
            "Apical Lateral": list(),
            "Lateral": list(),
            "Basal Lateral": list()
        }

        # 中值濾波需要的資訊
        self.AfterBlur = {
            "Apical Septal": list(),
            "Septal": list(),
            "Basal Septal": list(),
            "Apical Lateral": list(),
            "Lateral": list(),
            "Basal Lateral": list()
        }

    def _MedBlur(self, Left, Right):
        length = len(Left)

        for i in range(length):
            if i == 0 or i == length - 1:
                self.LeftMValve.append(Left[i])
                self.RightMValve.append(Right[i])

            else:
                LeftPts = np.asarray(Left[i-1:i+2])
                RightPts = np.asarray(Right[i-1:i+2])

                # 取 x 和 y 軸個別的中值
                medLeftPt = tuple(np.sort(LeftPts, axis=0)[1])
                medRightPt = tuple(np.sort(RightPts, axis=0)[1])
                self.LeftMValve.append(medLeftPt)
                self.RightMValve.append(medRightPt)

    def MuscleMatching(self, LeftMValvePos, RightMValvePos, isOutputVideo=True):
        print(f'----- 正在處理 {self.Path} Matching -----')

        video = self.video
        maskROI = self.roi
        ox, oy = self.ox, self.oy
        frame_count = 0
        matching_list = list()

        self._MedBlur(LeftMValvePos, RightMValvePos)
        LeftMV, RightMV = self.LeftMValve, self.RightMValve

        while 1:
            ret, frame = video.read()

            if not ret:
                break

            # - 瓣膜位置 & 預處理
            LeftMVPts, RightMVPts = LeftMV[frame_count], RightMV[frame_count]
            frame[maskROI != 255] = [0, 0, 0]
            frame_cp = frame.copy()
            frame_count += 1
            # - 瓣膜位置 & 預處理 End.

            # -- MultiThreshold v3
            Multi = Mt.MultiThres(frame, maskROI, 4, 0, 255)
            Multi.SearchMax()
            MultiFrame = Multi.threshold()
            # -- MultiThreshold v3 End.

            # -- First Scope
            # Scope Threshold
            level = 1
            MTFrame = MultiFrame.copy()
            GrayLevel = np.unique(MultiFrame)
            level_thres = GrayLevel[level]
            MTFrame[MTFrame <= level_thres] = 0
            # End.

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            erosion = cv2.erode(MTFrame, kernel, iterations=3)  # 消除邊界白色點

            CntMulti, _ = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            ArrayCntMulti = Vp.SplitContour(CntMulti)

            # 在第一次的 Hull 加上 ROI 的中心點
            ArrayCntMulti = np.vstack((ArrayCntMulti, [[[ox, oy]]]))
            HullMulti = cv2.convexHull(ArrayCntMulti)
            # -- First Scope End.

            # -- 調整矩形範圍和模型的比例
            # 擬合矩形框 <- frame 和 MultiFrame 的範圍
            rect = cv2.boundingRect(HullMulti)
            x, y, w, h = rect

            # 處理模型縮放比例及範圍 <- 盡可能避免模型縮放時有部分區域落在 ROI 外
            maskRange = np.ones((h, w), np.uint8) * 255

            roi_inv = cv2.bitwise_xor(maskROI[y:y + h, x:x + w], maskRange)
            CntROI_inv, _ = cv2.findContours(roi_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            regOutY = list()
            for RoiOut in CntROI_inv:
                area = cv2.contourArea(RoiOut)
                if area > 10:
                    HullOut = cv2.moments(RoiOut)
                    OutX, OutY = int(HullOut["m10"] / HullOut["m00"]), int(HullOut["m01"] / HullOut["m00"])
                    regOutY.append(OutY)
            if len(regOutY) != 0:
                minY = regOutY[regOutY.index(min(regOutY))]
            else:
                minY = 0
            A4CInfo = A4CModel(w, h - minY)  # A4C Information, dict
            frame_target = frame_cp[y + minY:y + h, x:x + w]
            MT_target = MultiFrame[y + minY:y + h, x:x + w]

            # 調整瓣膜支點的縮放比例
            ReLeftMVx, ReLeftMVy = LeftMVPts[0] - x, LeftMVPts[1] - y - minY
            ReRightMVx, ReRightMVy = RightMVPts[0] - x, RightMVPts[1] - y - minY
            # -- 調整矩形範圍和模型的比例 End.

            # -- Matching LV Region
            ModelLVCenters = A4CInfo["LVModel"]["Centers"]
            ModelLVContours = A4CInfo["LVModel"]["Contours"]
            LVModel = A4CInfo["LVModel"]["src"]
            ModelHeight, ModelWidth = LVModel.shape

            # - 調整模型 matching 的初始位置 & Matching
            # 1. 模型 y axis 不得低於瓣膜位置
            modelLowestY = np.max(Vp.SplitContour(ModelLVContours[2])[:, 1])
            adjY = modelLowestY - ReRightMVy if ReRightMVy < modelLowestY else 0
            Apex_newY = ModelLVCenters[0][1] - adjY
            Apex_newX = (ReLeftMVx + ReRightMVx) // 2

            # 2. 假設 模型 Apex Center 的 x axis 會在瓣膜的中間
            # # 判斷瓣膜右側支點的 x axis 是否向右側偏 (會超過 target 的範圍)
            # print(f'frame count: {frame_count}')
            if RightMVPts[0] > x + w:
                # # (2) 瓣膜右側支點的 x axis 歪掉(向右側偏)
                # Current Handle: 不移動 x axis 只移動高度
                initMat = np.array([[1, 0, 0], [0, 1, -adjY]], np.float32)

            else:
                # # (1) 瓣膜的左右支點皆接近真實的位置
                x0, y0 = ModelWidth / 2,  ModelHeight / 2
                r = np.sqrt((Apex_newX - x0) ** 2 + (Apex_newY - y0) ** 2)
                theta = np.arccos((Apex_newX - x0) / r) * 180 / np.pi
                alpha = np.arccos((ModelLVCenters[0][0] - x0) / r) * 180 / np.pi
                angle = theta - alpha
                initMat = cv2.getRotationMatrix2D((ModelWidth / 2,  ModelHeight / 2), angle, 1)

            initModel = cv2.warpAffine(LVModel, initMat, (ModelWidth, ModelHeight))

            # 3. 設定 Matching 時的 vertical, horizontal, theta 參數 & Matching
            Vert = Apex_newY // 4
            VertRange = range(-Vert, 2 * Vert, 3 * Vert // 5)

            if RightMVPts[0] <= x + w:
                # # (1) 瓣膜的左右支點皆接近真實的位置
                Hori = int((ReRightMVx - ReLeftMVx) * 0.1)
                HoriRange = range(-Hori, Hori, 4)
                ThetaRange = range(0, 20, 5)

            else:
                # # (2) 瓣膜右側支點的 x axis 歪掉(向右側偏)
                # Current method: 使用第一版的參數(因為只有調整模型的高度並沒有調整角度, 第一版參數值還可以接受)
                HoriRange = range(-30, 10, 5)
                ThetaRange = range(0, 40, 5)

            FitLVModel, BestLVResult, BestLVParam = ModelMatching(
                src=MT_target,
                target=initModel,
                vertical=VertRange,
                horizontal=HoriRange,
                theta=ThetaRange
            )
            # print(f'{frame_count}, BestLVParam: {BestLVParam}')
            # - 調整模型 matching 的初始位置 & Matching End.

            # - 由於經過平移, 旋轉, 因此要修正原本在 Info 裡面的資訊
            AfterLVCnt, _ = cv2.findContours(FitLVModel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            BeforeLVCnt = A4CInfo["LVModel"]["Contours"]
            swapPos = list()

            for after_i, after in enumerate(AfterLVCnt):
                regSimilar = list()
                afterM = cv2.moments(after)
                afterCx, afterCy = int(afterM["m10"] / afterM["m00"]), int(afterM["m01"] / afterM["m00"])

                for before_i, before in enumerate(BeforeLVCnt):
                    similar = cv2.matchShapes(before, after, cv2.CONTOURS_MATCH_I1, 0.0)
                    regSimilar.append(similar)

                minimum = regSimilar.index(min(regSimilar))
                swapPos.append([after_i, minimum, [afterCx, afterCy]])

            for update_i, update in enumerate(swapPos):
                new, old, cen = update
                A4CInfo["LVModel"]["Contours"][old] = AfterLVCnt[new]
                A4CInfo["LVModel"]["Centers"][old] = cen
            A4CInfo["LVModel"]["src"] = FitLVModel
            # - 由於經過平移, 旋轉, 因此要修正原本在 Info 裡面的資訊 End.

            # Sampling
            Connect = Vp.ConnectBound(MT_target, maskROI[y + minY:y + h, x:x + w])
            Connect.ContourSampling(step=8)
            SamplingInfo = Connect.SamplingInfo
            # End.

            # - 找出 Multi-Threshold 在各區段的所有點 + Sampling 的點做 曲線擬合
            LVRegionCnt = A4CInfo["LVModel"]["Contours"]
            TotalFitPts = list()
            funcList = list()

            for curr, cnt_lv in enumerate(LVRegionCnt):
                currFit = list()
                func = None

                # Sampling 的點
                internalPts = list()

                for key in SamplingInfo.keys():
                    sample_pts = SamplingInfo[key]["Sampling"]

                    if len(sample_pts) != 0:
                        for k in sample_pts:
                            dist = cv2.pointPolygonTest(cnt_lv, tuple(k), measureDist=True)
                            if dist >= -5:
                                internalPts.append(k)
                                # cv2.circle(frame_target, tuple(k), 3, (0, 0, 255), -1)

                internalPts = np.asarray(internalPts)
                if len(internalPts) != 0:
                    OriX1, OriY1 = internalPts[:, 0], internalPts[:, 1]
                else:
                    OriX1, OriY1 = np.array([], np.int), np.array([], np.int)
                # End.

                # Multi threshold 的點
                regMask = np.zeros(MT_target.shape, np.uint8)
                cv2.drawContours(regMask, [cnt_lv], -1, (255, 255, 255), -1)

                MTRange = cv2.bitwise_and(regMask, MT_target)
                MTRange[MTRange != 0] = 255

                OriY2, OriX2 = np.argwhere(MTRange != 0).T
                # End.

                OriY, OriX = np.hstack([OriY1, OriY2]), np.hstack([OriX1, OriX2])

                if curr == 0:  # apex of the heart
                    XData = np.unique(OriX)
                    YData = np.zeros(XData.shape, np.int)

                    for ind, uni_x in enumerate(XData):
                        YData[ind] = np.mean(OriY[OriX == uni_x]).astype(np.int)

                    func = np.polyfit(XData, YData, 2)
                    func = np.poly1d(func)
                    newY = func(XData).astype(np.int)
                    currFit = np.vstack([XData, newY]).T

                elif curr == 1:  # Septal
                    YData = np.unique(OriY)
                    XData = np.zeros(YData.shape, np.int)

                    for ind, uni_y in enumerate(YData):
                        XData[ind] = np.mean(OriX[OriY == uni_y]).astype(np.int)

                    func = np.polyfit(YData, XData, 1)
                    func = np.poly1d(func)
                    newX = func(YData).astype(np.int)
                    currFit = np.vstack([newX, YData]).T

                elif curr == 2:  # Lateral
                    YData = np.unique(OriY)
                    XData = np.zeros(YData.shape, np.int)

                    for ind, uni_y in enumerate(YData):
                        XData[ind] = np.mean(OriX[OriY == uni_y]).astype(np.int)

                    func = np.polyfit(YData, XData, 2)
                    func = np.poly1d(func)
                    newX = func(YData).astype(np.int)
                    currFit = np.vstack([newX, YData]).T

                TotalFitPts.append(currFit)
                funcList.append(func)
            # - 找出 Multi-Threshold 在各區段的所有點 + Sampling 的點做 曲線擬合 End.

            # - 修正超出範圍的點, 和擬合三段曲線
            regLeftFitMask = np.zeros(MT_target.shape, np.uint8)
            regRightFitMask = np.zeros(MT_target.shape, np.uint8)

            # Q1. apex of the heart 會有超過範圍的情況(左)
            # Sol1. 利用 Septal 垂直向上畫一刀
            regInd = np.argmin(TotalFitPts[1][:, 1])
            SeptalPt = TotalFitPts[1][regInd]
            apexLeftPts = TotalFitPts[0][TotalFitPts[0][:, 0] >= SeptalPt[0]]
            TotalFitPts[0] = apexLeftPts

            # 將線連接在一起
            apexLeftPt = apexLeftPts[0]
            # cv2.line(frame_target, tuple(apexLeftPt), tuple(SeptalPt), (255, 0, 0), 2)
            cv2.line(regLeftFitMask, tuple(apexLeftPt), tuple(SeptalPt), (255, 255, 255), 2)
            # End.

            # Q2. apex of the heart 會有超過範圍的情況(右)  <- 暫時直接用一條線連接, 不考慮取線交點的問題
            # Q4. Lateral 和 Right MV 的點連線
            ReRightMVx, ReRightMVy = RightMVPts[0] - x, RightMVPts[1] - y - minY  # 調整右側瓣膜支點的縮放比例

            # current Sol2.
            apexRightPt = TotalFitPts[0][-1]
            lateralUpPt = TotalFitPts[2][0]
            # cv2.line(frame_target, tuple(apexRightPt), tuple(lateralUpPt), (255, 0, 0), 2)
            cv2.line(regRightFitMask, tuple(apexRightPt), tuple(lateralUpPt), (255, 255, 255), 2)

            lateralLowPt = TotalFitPts[2][-1]
            # cv2.line(frame_target, (ReRightMVx, ReRightMVy), tuple(lateralLowPt), (255, 0, 0), 2)
            cv2.line(regRightFitMask, (ReRightMVx, ReRightMVy), tuple(lateralLowPt), (255, 255, 255), 2)

            # Q3. Septal 和 Left MV 的點連線 <- 縮放比例的問題
            SeptalPts = TotalFitPts[1][-1]
            ReLeftMVx, ReLeftMVy = LeftMVPts[0] - x, LeftMVPts[1] - y - minY
            # cv2.line(frame_target, tuple(SeptalPts), (ReLeftMVx, ReLeftMVy), (255, 0, 0), 2)
            cv2.line(regLeftFitMask, tuple(SeptalPts), (ReLeftMVx, ReLeftMVy), (255, 255, 255), 2)
            # End.
            # - 修正超出範圍的點, 和擬合三段曲線 End.

            # - 將三段的點 重新擬合
            # 只要暫時先處理第一個 case 即可
            ApexPoint = TotalFitPts[0][np.argmin(TotalFitPts[0][:, 1])]  # 取出 Apex Point
            # print(f'ApexPoint: {ApexPoint}')

            # Left points
            _ApexLeft = TotalFitPts[0][TotalFitPts[0][:, 0] <= ApexPoint[0]]
            LeftLinePts = np.argwhere(regLeftFitMask == 255)  # [y, x]
            _Septal = TotalFitPts[1]
            LeftOriX = np.hstack([_ApexLeft[:, 0], LeftLinePts[:, 1], _Septal[:, 0]])
            LeftOriY = np.hstack([_ApexLeft[:, 1], LeftLinePts[:, 0], _Septal[:, 1]])

            # Right points
            _ApexRight = TotalFitPts[0][TotalFitPts[0][:, 0] >= ApexPoint[0]]
            RightLinePts = np.argwhere(regRightFitMask == 255)  # [y, x]
            _lateral = TotalFitPts[2]
            RightOriX = np.hstack([_ApexRight[:, 0], RightLinePts[:, 1], _lateral[:, 0]])
            RightOriY = np.hstack([_ApexRight[:, 1], RightLinePts[:, 0], _lateral[:, 1]])

            # Left 和 Right 兩段皆用 y 預測 x
            LeftYData = np.unique(LeftOriY)
            LeftXData = np.zeros(LeftYData.shape, np.int)

            for ind, uni_y in enumerate(LeftYData):
                LeftXData[ind] = np.mean(LeftOriX[LeftOriY == uni_y]).astype(np.int)

            func = np.polyfit(LeftYData, LeftXData, 5)
            func = np.poly1d(func)
            LeftYData = np.arange(np.min(LeftYData), np.max(LeftYData) + 1)

            newLeftX = func(LeftYData).astype(np.int)
            LeftFit = np.vstack([newLeftX, LeftYData]).T

            RightYData = np.unique(RightOriY)
            RightXData = np.zeros(RightYData.shape, np.int)

            for ind, uni_y in enumerate(RightYData):
                RightXData[ind] = np.mean(RightOriX[RightOriY == uni_y]).astype(np.int)

            func = np.polyfit(RightYData, RightXData, 6)
            func = np.poly1d(func)
            RightYData = np.arange(np.min(RightYData), np.max(RightYData) + 1)
            newRightX = func(RightYData).astype(np.int)
            RightFit = np.vstack([newRightX, RightYData]).T

            # 重新和瓣膜連線
            # cv2.line(frame_target, tuple(LeftFit[-1]), (ReLeftMVx, ReLeftMVy), (255, 0, 0), 2)
            # cv2.line(frame_target, tuple(RightFit[-1]), (ReRightMVx, ReRightMVy), (255, 0, 0), 2)
            # - 將三段的點 重新擬合 End.

            # - 三段點重新取樣 (以長度區分)
            # colors = [(0, 255, 255), (0, 255, 0), (255, 255, 0)]
            LeftLength, RightLength = 0, 0

            # 重新取樣的擬合線
            for l_index, Pt in enumerate(LeftFit):
                if l_index > 0:
                    regLx, regLy = LeftFit[l_index - 1]
                    # display
                    # cv2.line(frame_target, tuple(Pt), tuple(LeftFit[l_index - 1]), colors[0], 2)
                    # 計算長度
                    LeftLength += np.sqrt((Pt[0] - regLx) ** 2 + (Pt[1] - regLy) ** 2)
                else:
                    # display
                    # cv2.line(frame_target, tuple(Pt), tuple(ApexPoint), colors[0], 2)
                    pass

            for r_index, Pt in enumerate(RightFit):
                if r_index > 0:
                    regRx, regRy = RightFit[r_index - 1]
                    # display
                    # cv2.line(frame_target, tuple(Pt), tuple(RightFit[r_index - 1]), colors[1], 2)
                    # 計算長度
                    RightLength += np.sqrt((Pt[0] - regRx) ** 2 + (Pt[1] - regRy) ** 2)
                else:
                    # display
                    # cv2.line(frame_target, tuple(Pt), tuple(ApexPoint), colors[1], 2)
                    pass
            # End.

            # 左邊肌肉取樣
            muscle_semantic = self.MuscleSemantic
            LeftLength = int(LeftLength) + 1
            LeftEachDis = LeftLength // 14

            LeftColors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
            len_L, color_i, pos_cnt = 0, 0, 0
            regLeftPt = list()

            for l_index, Pt in enumerate(LeftFit):
                if l_index > 0:
                    regLx, regLy = LeftFit[l_index - 1]
                    len_L += np.sqrt((Pt[0] - regLx) ** 2 + (Pt[1] - regLy) ** 2)

                    if len_L >= LeftEachDis:
                        len_L = len_L - LeftEachDis
                        pos_cnt += 1

                        color_i = color_i + 1 if pos_cnt % 5 == 0 else color_i
                        rad = 5 if pos_cnt % 5 == 2 else 3
                        # cv2.circle(frame_target, tuple(Pt), rad, LeftColors[color_i], -1)
                        regLeftPt.append(list(Pt))

                    elif l_index == len(LeftFit) - 1 and pos_cnt == 13:
                        # cv2.circle(frame_target, tuple(Pt), 3, LeftColors[color_i], -1)
                        regLeftPt.append(list(Pt))

                else:
                    # cv2.circle(frame_target, tuple(Pt), 3, LeftColors[color_i], -1)
                    regLeftPt.append(list(Pt))

                if len(regLeftPt) == 5:
                    muscle_semantic[list(muscle_semantic.keys())[color_i]].append(regLeftPt)
                    regLeftPt = list()
            # End.

            # 右邊肌肉取樣
            RightLength = int(RightLength) + 1
            RightEachDis = RightLength // 14
            RightColors = [(255, 255, 0), (0, 255, 255), (255, 0, 255)]
            len_R, color_i, pos_cnt = 0, 0, 0
            regRightPt = list()

            for r_index, Pt in enumerate(RightFit):
                if r_index > 0:
                    regRx, regRy = RightFit[r_index - 1]
                    len_R += np.sqrt((Pt[0] - regRx) ** 2 + (Pt[1] - regRy) ** 2)

                    if len_R >= RightEachDis:
                        len_R = len_R - RightEachDis
                        pos_cnt += 1

                        color_i = color_i + 1 if pos_cnt % 5 == 0 else color_i
                        rad = 5 if pos_cnt % 5 == 2 else 3
                        # cv2.circle(frame_target, tuple(Pt), rad, RightColors[color_i], -1)
                        regRightPt.append(list(Pt))

                    elif r_index == len(RightFit) - 1 and pos_cnt == 13:
                        # cv2.circle(frame_target, tuple(Pt), 3, RightColors[color_i], -1)
                        regRightPt.append(list(Pt))

                else:
                    # cv2.circle(frame_target, tuple(Pt), 3, RightColors[color_i], -1)
                    regRightPt.append(list(Pt))

                if len(regRightPt) == 5:
                    muscle_semantic[list(muscle_semantic.keys())[color_i + 3]].append(regRightPt)
                    regRightPt = list()

            # cv2.putText(frame_cp, f'frame_count: {frame_count}', (40, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
            #             (255, 255, 255), 1)
            #
            # FitLVModel = cv2.cvtColor(FitLVModel, cv2.COLOR_GRAY2BGR)
            # check = cv2.addWeighted(frame_target, 1, FitLVModel, 0.6, 1)
            # cv2.imshow('check', check)
            #
            # cv2.imshow('frame_target', frame_target)
            # cv2.imshow('LV Model', LVModel)
            # cv2.imshow('MT', MT_target)
            # cv2.imshow('BestLVResult', BestLVResult)
            # cv2.imshow('initModel', initModel)
            # cv2.imshow('FitLVModel', FitLVModel)
            #
            # if cv2.waitKey(0) == ord('n'):
            #     continue
            # elif cv2.waitKey(0) == ord('q'):
            #     break
            # matching_list.append(frame_cp)

        # (Muscle Semantic Sample Points) 中值濾波每個點的結果
        for key in self.MuscleSemantic.keys():
            array = np.asarray(self.MuscleSemantic[key])
            TotalFrame, PtPos, Pt = array.shape

            for i in range(TotalFrame):
                regBlurPt = list()
                for Pos in range(PtPos):
                    if i == 0 or i == TotalFrame - 1:
                        regBlurPt.append(array[:, Pos][i])
                    else:
                        after_blur = np.sort(array[:, Pos][i-1:i+2], axis=0)[1]
                        regBlurPt.append(list(after_blur))
                self.AfterBlur[key].append(regBlurPt)
        # print(f'After Blur: {self.AfterBlur["Apical Septal"]}')
        # End.

        # 重新對應每一幀畫上濾波後的點
        frame_count = 0
        Colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        video = cv2.VideoCapture(self.Path)
        while 1:
            ret, frame = video.read()

            if not ret:
                break

            # 原本的影像預處理
            frame_cp = frame.copy()
            frame[maskROI != 255] = [0, 0, 0]
            # - 瓣膜位置 & 預處理 End.

            # -- MultiThreshold v3
            Multi = Mt.MultiThres(frame, maskROI, 4, 0, 255)
            Multi.SearchMax()
            MultiFrame = Multi.threshold()
            # -- MultiThreshold v3 End.

            # -- First Scope
            # Scope Threshold
            level = 1
            MTFrame = MultiFrame.copy()
            GrayLevel = np.unique(MultiFrame)
            level_thres = GrayLevel[level]
            MTFrame[MTFrame <= level_thres] = 0
            # End.

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            erosion = cv2.erode(MTFrame, kernel, iterations=3)  # 消除邊界白色點

            CntMulti, _ = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            ArrayCntMulti = Vp.SplitContour(CntMulti)

            # 在第一次的 Hull 加上 ROI 的中心點
            ArrayCntMulti = np.vstack((ArrayCntMulti, [[[ox, oy]]]))
            HullMulti = cv2.convexHull(ArrayCntMulti)
            # -- First Scope End.

            # -- 調整矩形範圍和模型的比例
            # 擬合矩形框 <- frame 和 MultiFrame 的範圍
            rect = cv2.boundingRect(HullMulti)
            x, y, w, h = rect

            # 處理模型縮放比例及範圍 <- 盡可能避免模型縮放時有部分區域落在 ROI 外
            maskRange = np.ones((h, w), np.uint8) * 255

            roi_inv = cv2.bitwise_xor(maskROI[y:y + h, x:x + w], maskRange)
            CntROI_inv, _ = cv2.findContours(roi_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            regOutY = list()
            for RoiOut in CntROI_inv:
                area = cv2.contourArea(RoiOut)
                if area > 10:
                    HullOut = cv2.moments(RoiOut)
                    OutX, OutY = int(HullOut["m10"] / HullOut["m00"]), int(HullOut["m01"] / HullOut["m00"])
                    regOutY.append(OutY)
            if len(regOutY) != 0:
                minY = regOutY[regOutY.index(min(regOutY))]
            else:
                minY = 0

            frame_target = frame_cp[y + minY:y + h, x:x + w]
            # 處理模型縮放比例及範圍 <- 盡可能避免模型縮放時有部分區域落在 ROI 外

            a = self.AfterBlur["Apical Septal"][frame_count]
            b = self.AfterBlur["Septal"][frame_count]
            c = self.AfterBlur["Basal Septal"][frame_count]
            d = self.AfterBlur["Apical Lateral"][frame_count]
            e = self.AfterBlur["Lateral"][frame_count]
            f = self.AfterBlur["Basal Lateral"][frame_count]

            curr_pts = [a, b, c, d, e, f]
            for i in range(6):
                for j in range(5):
                    rad = 5 if j == 2 else 3
                    cv2.circle(frame_target, tuple(curr_pts[i][j]), rad, Colors[i], -1)

            frame_count += 1
            cv2.putText(frame_cp, f'frame_count: {frame_count}', (60, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        (255, 255, 255), 1)

            cv2.putText(frame_cp, 'Apical Septal', (60, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7,
                        (0, 255, 0), 1)
            cv2.putText(frame_cp, 'Septal', (60, 140), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7,
                        (255, 0, 0), 1)
            cv2.putText(frame_cp, 'Basal Septal', (60, 160), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7,
                        (0, 0, 255), 1)
            cv2.putText(frame_cp, 'Apical Lateral', (60, 180), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7,
                        (255, 255, 0), 1)
            cv2.putText(frame_cp, 'Lateral', (60, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7,
                        (0, 255, 255), 1)
            cv2.putText(frame_cp, 'Basal Lateral', (60, 220), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7,
                        (255, 0, 255), 1)
            frame_cp[y + minY:y + h, x:x + w] = frame_target

            matching_list.append(frame_cp)

        if not os.path.isdir(self.OutputMatchingDir):
            os.makedirs(self.OutputMatchingDir)

        if isOutputVideo:
            FileName = self.Path.split('\\')[-1]
            FileIO.write_video(matching_list, self.OutputMatchingDir + ' Match v2_GLSPt ' + FileName)
