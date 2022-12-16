import cv2
import numpy as np


def ImageContrast(img, brightness=0, contrast=0):
    """
    ImageContrast(img, brightness=0, contrast=0):
        調整影像對比度 & 亮度

    parameters:
        img: 灰階圖像
        brightness: 亮度, 值越高整張圖片亮度越大否則反之, 默認 0
        contrast: 對比度, > 0 則對比越強烈, < 0 對比較不清晰, 默認 0
    """
    B = brightness / 255.0
    C = contrast / 255.0
    k = np.tan((45 + 44 * C) / 180 * np.pi)

    img = (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def SplitContour(Cnts):
    """
    function: SplitContourL(Cnts):
        拆分
    """
    CntList = list()
    for c1 in range(len(Cnts)):
        for c2 in range(len(Cnts[c1])):
            CntList.append(Cnts[c1][c2])
    CntList = np.asarray(CntList)
    return CntList


class ConnectBound(object):
    def __init__(self, src, mask, LevelThres=1):
        """
        parameters:
            src: 輸入影像(灰階 or 彩色皆可)
            mask: 要計算的 ROI 區域 (二值圖)
            LevelThres: Multi-Threshold 的階數, int
        """
        self.src = src
        self.mask = mask
        self.level = LevelThres

        self.handle = self._HandleMTsrc(LevelThres)
        self.speckle = np.zeros(src.shape, np.uint8)

        self.SamplingInfo = dict()
        self.PosList = list()

    def _HandleMTsrc(self, LevelThres=1):
        img = self.src
        img[self.mask != 255] = 0

        # -- handle gray level value
        GrayLevel = np.unique(img)
        if LevelThres >= len(GrayLevel) - 1:
            raise ValueError('LevelThres 灰階門檻值的數值大於 Multi-Threshold 的階數 (MuscleSampling.py)')

        LevelThreshold = GrayLevel[LevelThres]
        img[img <= LevelThreshold] = 0
        # -- handle gray level value End.

        return img

    def _Sampling(self, Cnt, Pos, step):
        CenterM = cv2.moments(Cnt)
        Cx, Cy = int(CenterM["m10"] / CenterM["m00"]), int(CenterM["m01"] / CenterM["m00"])
        self.SamplingInfo[Pos]["Center"] = (Cx, Cy)
        self.SamplingInfo[Pos]["Sampling"] = list()

        rect = cv2.minAreaRect(Cnt)
        box = cv2.boxPoints(rect).astype(np.int)

        # - 利用 L2 distance 定義四點位置
        px0, py0 = box[0, :]
        d = np.sqrt((px0 - box[1:, 0]) ** 2 + (py0 - box[1:, 1]) ** 2)
        min_pos, max_pos = int(np.argmin(d)) + 1, int(np.argmax(d)) + 1

        other = ({1, 2, 3} - {min_pos, max_pos}).pop()
        px1, py1 = box[min_pos, :]
        px2, py2 = box[max_pos, :]
        px3, py3 = box[other, :]
        # - 利用 L2 distance 定義四點位置 End.

        # - 根據四點位置判斷長寬(x axis 代表 length, y axis 代表 width)
        # 目的: 長邊畫點, 短邊畫線
        deltaY01, deltaX01 = abs(py1 - py0), abs(px1 - px0)
        # deltaY 小(x axis) 取 y 軸的點

        if deltaY01 > deltaX01:
            # width: [(p0, p1), (p2, p3)], length: [(p0, p3), (p1, p2)]

            # -- 處理 box 的 float 轉 int 導致兩邊長度不同
            fill_x1, fill_x2 = 0, 0
            isEqualX = abs(px0 - px3) - abs(px1 - px2)
            if isEqualX > 0:
                fill_x1 = -isEqualX
            elif isEqualX < 0:
                fill_x2 = isEqualX
            # -- End.

            # -- 處理 X, Y 的點, 並連線
            if abs(px0 - px3) + fill_x1 != 0:
                X1pts = np.arange(min(px0, px3), max(px0, px3) + fill_x1 + 1, step)
                X2pts = np.arange(min(px1, px2), max(px1, px2) + fill_x2 + 1, step)

                # 兩線平行: slope 相同
                slope = (py3 - py0) / (px3 - px0)
                b1 = int(py0 - slope * px0)
                b2 = int(py1 - slope * px1)
                Y1pts = (X1pts * slope + b1).astype(np.int)
                Y2pts = (X2pts * slope + b2).astype(np.int)

            else:
                # slope = infinite
                X1pts = np.arange(min(px0, px3), max(px0, px3) + fill_x1 + 1, step)
                X2pts = np.arange(min(px1, px2), max(px1, px2) + fill_x2 + 1, step)
                Y1pts = np.ones(X1pts.shape, np.int) * py0
                Y2pts = np.ones(X2pts.shape, np.int) * py1
            # -- 處理 X, Y 的點, 並連線 End.

        else:
            # width: [(p0, p3), (p1, p2)], length: [(p0, p1), (p2, p3)]
            # -- 處理 box 的 float 轉 int 導致兩邊長度不同
            fill_y1, fill_y2 = 0, 0
            isEqualY = abs(py0 - py3) - abs(py1 - py2)
            if isEqualY > 0:
                fill_y1 = -isEqualY
            elif isEqualY < 0:
                fill_y2 = isEqualY
            # -- End.

            # -- 處理 X, Y 的點, 並連線
            if abs(py0 - py3) + fill_y1 != 0:
                Y1pts = np.arange(min(py0, py3), max(py0, py3) + fill_y1 + 1, step)
                Y2pts = np.arange(min(py1, py2), max(py1, py2) + fill_y2 + 1, step)

                # 兩線平行: slope 相同
                slope = (px3 - px0) / (py3 - py0)
                b1 = int(px0 - slope * py0)
                b2 = int(px1 - slope * py1)
                X1pts = (Y1pts * slope + b1).astype(np.int)
                X2pts = (Y2pts * slope + b2).astype(np.int)

            else:
                # slope = infinite
                Y1pts = np.arange(min(py0, py3), max(py0, py3) + fill_y1 + 1, step)
                Y2pts = np.arange(min(py1, py2), max(py1, py2) + fill_y2 + 1, step)
                X1pts = np.ones(Y1pts.shape, np.int) * px0
                X2pts = np.ones(Y2pts.shape, np.int) * px2

            # -- 處理 X, Y 的點, 並連線 End.
        # - 根據四點位置判斷長寬 End.

        # - 切分成 n 矩形後, 找出中心點及畫線
        mask_sample = np.zeros(self.src.shape, np.uint8)
        cv2.drawContours(mask_sample, [Cnt], -1, (255, 255, 255), -1)

        targetPts = list()
        for i in range(1, len(X1pts)):
            mask_locate = np.zeros(mask_sample.shape, np.uint8)
            maskRect = np.array([
                [X1pts[i-1], Y1pts[i-1]], [X2pts[i-1], Y2pts[i-1]],
                [X2pts[i], Y2pts[i]], [X1pts[i], Y1pts[i]]
                 ])
            cv2.drawContours(mask_locate, [maskRect], -1, (255, 255, 255), -1)

            target = cv2.bitwise_and(mask_locate, mask_sample)
            targetCnt, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # - 若輪廓數量 = 0, 則跳過
            if len(targetCnt) == 0:
                continue

            # - 畫中心點; 若輪廓面積 = 0, 跳過
            for ii in range(len(targetCnt)):
                if cv2.contourArea(targetCnt[ii]) > 0:
                    MSample = cv2.moments(targetCnt[ii])
                    SampleCx, SampleCy = int(MSample["m10"] / MSample["m00"]), int(MSample["m01"] / MSample["m00"])
                    self.SamplingInfo[Pos]["Sampling"].append([SampleCx, SampleCy])

                    # - 畫中心點
                    # cv2.circle(self.display, (SampleCx, SampleCy), 3, (0, 0, 255), -1)
                    cv2.circle(self.speckle, (SampleCx, SampleCy), 5, (255, 255, 255), -1)
                    # - End.
                    targetPts.append([SampleCx, SampleCy])
            # - 畫中心點; 若輪廓面積 = 0, 則使用輪廓的第一個點當成輪廓中心 End.

    def ContourSampling(self, step=8):
        # -- Contour Sampling
        Cnt, _ = cv2.findContours(self.handle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        PosList = self.PosList

        for pos, cnt in enumerate(Cnt):
            area = cv2.contourArea(cnt)
            if area > 10:
                self.SamplingInfo[pos] = dict()
                self._Sampling(cnt, pos, step)
                PosList.append(pos)
        # -- Contour Sampling End.

