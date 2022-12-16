from ctypes import *
import cv2
import numpy as np
from Cconvolution2 import *
import matplotlib.pyplot as plt


class MultiThres(object):
    def __init__(self, src, ROI, Level=3, MinThres=0, MaxThres=255):
        """
        parameters:
            src: 輸入原始圖像
            ROI: ROI 區域
            Level: MultiThreshold 的階數
            MinThres: 最小門檻值
            MaxThres: 最大門檻值
        """

        self.src = src
        self.roi = ROI
        self.level = Level
        self.interpolate = self._interpolate(ROI, MinThres, MaxThres)

        self.MinThres, self.MaxThres = MinThres, MaxThres
        self.interval = np.linspace(MinThres, MaxThres, Level + 1).astype(np.int)
        self.ValueList = [0] * (2 * Level + 1)

    def _interpolate(self, roi, MinThres, MaxThres):
        # --- 處理影像是否灰階
        if len(self.src.shape) > 2:
            self.src = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
        # --- 處理影像是否灰階 End.

        # --- 統計影像灰階直方圖, 去除非 ROI 區域
        self.src[self.roi != 255] = 0
        unique = np.unique(self.src, return_counts=True)
        unique_roi = np.unique(roi, return_counts=True)
        not_roi_effect = unique_roi[1][0] if len(unique_roi[0]) == 2 else 0
        unique[1][0] = unique[1][0] - not_roi_effect
        # --- 統計影像灰階直方圖, 去除非 ROI 區域 End.

        # --- 一階內插
        # 1. 處理直方圖對應灰階值索引
        hist = np.zeros(256, np.int)
        for index, scale in enumerate(unique[0]):
            hist[scale] = unique[1][index]
        # 1. 處理直方圖對應灰階值索引 End.

        # 2. 線性內插
        for index, scale in enumerate(unique[0]):
            if index == len(unique[0]) - 1:
                break

            if MinThres <= scale <= MaxThres:
                # 代表灰階間有缺值
                if unique[0][index + 1] - unique[0][index] > 1:
                    x_interval = unique[0][index + 1] - unique[0][index]
                    y_interval = unique[1][index + 1] - unique[1][index]
                    slope = y_interval / x_interval

                    for med in range(unique[0][index] + 1, unique[0][index + 1], 1):
                        y_insert = int(np.round(slope * (med - unique[0][index]) + unique[1][index], 0))
                        hist[med] = y_insert

        # 2. 線性內插 End.
        # --- 一階內插 End.
        return hist

    def SearchMax(self):
        """
        暫時先不考慮 delta 的部分 <- 在 3 階時, 差異沒有很明顯
        """

        hist = self.interpolate
        interval = self.interval

        # 這裡要對應給 C 語言創建 U 的參數, 所以改成符合 C 的寫法
        # -- 計算 level 區間的加權平均值(高點)
        AvgList = self.ValueList
        AvgList[0], AvgList[-1] = interval[0], interval[-1]

        for pos in range(self.level):
            left, right = interval[pos], interval[pos + 1]

            weight = np.arange(left, right)
            partHist = hist[left:right]

            try:
                avg = int(np.sum(weight * partHist) / np.sum(partHist))
            except ZeroDivisionError:
                avg = interval[pos]
            AvgList[2 * pos + 1] = avg
        # -- 計算 level 區間的加權平均值(高點) End.

        # -- 計算高點之間的加權平均值(低點)
        for pos in range(self.level - 1):
            left, right = AvgList[2 * pos + 1], AvgList[2 * (pos + 1) + 1]

            weight = np.arange(left, right)
            partHist = hist[left:right]

            try:
                avg = int(np.sum(weight * partHist) / np.sum(partHist))
            except ZeroDivisionError:
                avg = AvgList[pos] + 1  # avoid the value of alpha is 0.
            AvgList[2 * (pos + 1)] = avg
        # -- 計算高點之間的加權平均值(低點) End.

    def threshold(self):
        """
        暫時先不考慮 Scale 的部分
        """
        height, width = self.src.shape

        # -- 創建 U & UX Map  problem area
        # - python list 轉成 c pointer
        data = intArray(len(self.ValueList))

        for index in range(len(self.ValueList)):
            data.__setitem__(index, int(self.ValueList[index]))
        U(data, len(self.ValueList))
        # -- 創建 U & UX Map End.

        # -- C MultiThreshold
        CMultiThreshold(self.src.astype(np.int32))
        result = cvar.result

        addr_x = c_int * width
        addr_xy = addr_x * height
        result = np.array(addr_xy.from_address(int(result))).astype(np.uint8)
        result[self.roi != 255] = 0
        # -- C MultiThreshold End.

        return result
