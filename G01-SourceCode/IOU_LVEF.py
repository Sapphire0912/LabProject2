import FileIO

from scipy import fftpack
from scipy import signal
import numpy as np
import cv2

import warnings
warnings.filterwarnings("ignore")


class LVEFInfo(object):
    def __init__(self, UnitCM, Radius):
        """
        parameters:
            UnitCM: 超音波標準長度 (cm), int
            Radius: ROI 的半徑 (pixel), int
        """
        self.UnitCM = UnitCM
        self.Radius = Radius
        self.EDVFrame, self.ESVFrame = list(), list()
        self.CurrResult = dict()

    def FindCycle(self, CurrentResult, fps=None, BPM=None):
        """
        description:
            找出週期後, 定義 EDV 和 ESV 的時間, 並取出各 5 幀.
            e.g: EDV 在 frame count = 19 的位置, 則取出 [17, 18, 19, 20, 21] 的 frame, ESV 同理.

        method:
            FindCycle(self, CurrentResult[, fps=None[, BPM=None]]):
                找出週期以及 EDV 和 ESV 的時間.

        parameters:
            CurrentResult: 原本的 result_area 結果, 套上字典後的輸入.(format: {frame_count: result_area})
                           result_area 為灰階的 mask, frame_count, int.
            fps: frames per second, int. default: None.
            BPM: beats per minute, int. default: None.
            若 fps or BPM 其中一個是 None, 則採用傅立葉變換來找週期

        return:
            wait;
        """
        self.CurrResult = CurrentResult

        VolumeList = list()
        for i in CurrentResult.keys():
            cnt, _ = cv2.findContours(CurrentResult[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area = cv2.contourArea(cnt[0])
            VolumeList.append(area)

        if (not fps) or (not BPM):
            # 低通濾波
            y1 = signal.medfilt(VolumeList, 7)
            sig_fft = fftpack.fft(y1)
            power = np.abs(sig_fft) ** 2

            sample_freq = fftpack.fftfreq(len(y1), d=7)
            pos_mask = np.where(sample_freq > 0)
            freqs = sample_freq[pos_mask]
            peak_freq = freqs[power[pos_mask].argmax()]
            np.allclose(peak_freq, 1. / 40)
            high_freq_fft = sig_fft.copy()
            high_freq_fft[np.abs(sample_freq) > (peak_freq * 1.7)] = 0
            filtered_sig = fftpack.ifft(high_freq_fft)

            x = np.arange(1, len(VolumeList) + 1, 1)
            a = np.diff(np.sign(np.diff(filtered_sig))).nonzero()[0] + 1  # local min & max
            b = (np.diff(np.sign(np.diff(filtered_sig))) > 0).nonzero()[0] + 1  # local min
            c = (np.diff(np.sign(np.diff(filtered_sig))) < 0).nonzero()[0] + 1  # local max
            # +1 due to the fact that diff reduces the original index number

            EDVTime, ESVTime = c[0], b[0]

        else:
            CycleCount = round(60 * fps / BPM)  # type <int>
            target = VolumeList[:CycleCount]
            EDV, ESV = max(target), min(target)
            EDVTime, ESVTime = target.index(EDV), target.index(ESV)

        if 2 < EDVTime < len(CurrentResult) - 2:
            self.EDVFrame = [EDVTime + i for i in range(-2, 3)]
        elif EDVTime <= 2:
            self.EDVFrame = [1, 2, 3, 4, 5]
        else:
            maxFrame = len(CurrentResult)
            self.EDVFrame = [maxFrame - i for i in range(4, -1, -1)]

        if 2 < ESVTime < len(CurrentResult) - 2:
            self.ESVFrame = [ESVTime + i for i in range(-2, 3)]
        elif EDVTime <= 2:
            self.ESVFrame = [1, 2, 3, 4, 5]
        else:
            maxFrame = len(CurrentResult)
            self.ESVFrame = [maxFrame - i for i in range(4, -1, -1)]

    def calLVEF(self, mode='self', GTDir=None):
        EDV, ESV = self.EDVFrame, self.ESVFrame
        Text = ["Hyper Dynamic", "Normal", "Mild", "Moderate", "Severe"]
        Degree = np.array([70, 49, 39, 29, 0])

        if mode == 'self':
            frame_area = self.CurrResult
            EDVVolume = [self.calLVVolume(frame_area[i]) for i in EDV]
            ESVVolume = [self.calLVVolume(frame_area[i]) for i in ESV]

        elif mode == 'GT':
            if GTDir is None:
                raise ValueError("mode 為 GT 時, 必須輸入 GTDir(GroundTruth) 的資料夾路徑")

            GTFiles = FileIO.AllFiles(GTDir, 'png')
            GTDict = dict()
            for files in GTFiles:
                frame_count = int(files.split('_')[-1].split('.')[0])
                GTDict[frame_count] = files

            EDVVolume = [self.calLVVolume(cv2.imread(GTDict[i], 0)) for i in EDV]
            ESVVolume = [self.calLVVolume(cv2.imread(GTDict[i], 0)) for i in ESV]

        else:
            raise ValueError("mode 參數 輸入格式錯誤")

        maxEDV, minESV = max(EDVVolume), min(ESVVolume)
        LVEF = round(100 * (maxEDV - minESV) / maxEDV, 1)
        degree = Text[len(Degree[Degree > LVEF])]

        return maxEDV, minESV, LVEF, degree

    def calLVVolume(self, src):
        """
        parameters:
            src: 二值化圖像, np.ndarray.

        return:
            LVVolume: 左心室容積單位 cm^3(ml), float
        """

        mask_cnt = np.zeros(src.shape, np.uint8)
        mask_fill_cnt = np.zeros(src.shape, np.uint8)
        cnt, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnt) != 1:
            areaList = list()
            for c in cnt:
                areaList.append(cv2.contourArea(c))
            cnt_index = areaList.index(max(areaList))
            cnt_target = cnt[cnt_index]
        else:
            cnt_target = cnt[0]

        cv2.drawContours(mask_cnt, [cnt_target], -1, (255, 255, 255), 1)
        cv2.drawContours(mask_fill_cnt, [cnt_target], -1, (255, 255, 255), -1)
        cnt_target = cnt_target.reshape(len(cnt_target), -1)  # coordinate: [x, y]

        # find Apex point
        uni_yaxis = np.unique(cnt_target[:, 1])
        apexY = uni_yaxis[0]
        xaxis = cnt_target[cnt_target[:, 1] == apexY]  # 找出最低點(y 最小值)的 x 數值
        apexX = int(np.mean(xaxis[:, 0]))

        # find left & right point
        leftPoints, rightPoints = cnt_target[cnt_target[:, 0] < apexX], cnt_target[cnt_target[:, 0] >= apexX]
        uni_yaxis = np.unique(leftPoints[:, 1])
        leftY = uni_yaxis[-1]
        xaxis = leftPoints[leftPoints[:, 1] == leftY]
        leftX = xaxis[-1, 0]

        uni_yaxis = np.unique(rightPoints[:, 1])
        rightY = uni_yaxis[-1]
        xaxis = rightPoints[rightPoints[:, 1] == rightY]
        rightX = xaxis[-1, 0]

        # 垂直點 & 切 20 等分, 計算 LVEF (由瓣膜位置往上到 Apex Point)
        slope_lr = (rightY - leftY) / (rightX - leftX)
        bias_lr = rightY - slope_lr * rightX
        LVVolume = 0  # cm^3

        if slope_lr != 0:
            slope_apex = - (1 / slope_lr)
            bias_apex = apexY - slope_apex * apexX

            midX = (bias_apex - bias_lr) / (slope_lr - slope_apex)
            midY = slope_lr * midX + bias_lr
            midX, midY = int(midX), int(midY)
            h = np.sqrt((apexX - midX) ** 2 + (apexY - midY) ** 2)
            dh = h / 20

            for i in range(0, 19):
                mask_line = np.zeros(src.shape, np.uint8)

                bias_lr -= dh
                lineY = slope_lr * 800 + bias_lr
                cv2.line(mask_line, (0, int(bias_lr)), (800, int(lineY)), (255, 255, 255), 1)
                mask_line[mask_fill_cnt != 255] = 0

                cnt_line, _ = cv2.findContours(mask_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnt_line = cnt_line[0].reshape(len(cnt_line[0]), -1)  # coordinate: [x, y]

                # 計算 LVEF Volume
                minX, maxX = np.min(cnt_line[:, 0]),  np.max(cnt_line[:, 0])
                scale_cm = self.Radius / self.UnitCM

                r = ((maxX - minX) / 2) / scale_cm
                dh_scale = dh / scale_cm
                volume = r ** 2 * np.pi * dh_scale  # cm^3, 1cm^3 = 1ml
                LVVolume += volume

        else:
            dh = (leftY - apexY) / 20

            for i in range(0, 19):
                mask_line = np.zeros(src.shape, np.uint8)

                bias_lr -= dh
                cv2.line(mask_line, (0, int(bias_lr)), (800, int(bias_lr)), (255, 255, 255), 1)
                mask_line[mask_fill_cnt != 255] = 0

                cnt_line, _ = cv2.findContours(mask_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnt_line = cnt_line[0].reshape(len(cnt_line[0]), -1)  # coordinate: [x, y]

                # 計算 LVEF Volume
                minX, maxX = np.min(cnt_line[:, 0]),  np.max(cnt_line[:, 0])
                scale_cm = self.Radius / self.UnitCM

                r = ((maxX - minX) / 2) / scale_cm
                dh_scale = dh / scale_cm
                volume = r ** 2 * np.pi * dh_scale  # cm^3, 1cm^3 = 1ml
                LVVolume += volume

        return round(LVVolume, 1)
