import cv2
import numpy as np
import Rising_pt as Rs
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from MultiThreshold import MultiThres

# Target: 計算 LVEF


def TimeMedian(lst):
    """
    function:
        TimeMedian(lst): 將座標點做時間上的中值濾波

    parameters:
        lst: python list, 儲存每幀座標點(x, y), 為座標點為 tuple 類型.

    handle:
        A4CSegmentation 的輸出結果, 格式為每幀的座標(x, y), list(tuple).
        1. Mitral Valve 左側支點.
        2. Mitral Valve 右側支點.
        3. Left Ventricle 的中心點(Center), 每幀的座標(x, y), list(tuple).

    return:
        MedianPt: 時間濾波後的結果, 座標格式為 (x, y), list(tuple).
    """

    MedianPt = list()  # save result
    length = len(lst)

    array = np.asarray(lst)  # convert to np.array
    X_pts, Y_pts = array[:, 0], array[:, 1]

    for index in range(length):  # 倒數第 1, 2 個點不需濾波
        if index + 2 >= length:
            MedianPt.append(lst[-1])  # 倒數第 1, 2 個點等於原本的值

        else:
            Xpt = np.sort(X_pts[index:index+3])[1]  # median blur X point
            Ypt = np.sort(Y_pts[index:index+3])[1]  # median blur Y point
            MedianPt.append((Xpt, Ypt))

    return MedianPt


def dist_pt(a, b):
    x = np.array((int(a[0]), int(a[1])))
    y = np.array((int(b[0]), int(b[1])))
    dist = np.sqrt(np.sum(np.square(x - y)))
    return dist


def linear_LR(list_1, list_2, img, cen, power1, power2):
    x = []
    y = []
    for i in range(0, len(list_1), 1):
        x.append(int(list_1[i][1]))
        y.append(list_1[i][0])

    x_out = x
    x = np.array(x)
    y = np.array(y)
    x = x[:, np.newaxis]

    rx = []
    ry = []

    for i in range(0, len(list_2), 1):
        rx.append(int(list_2[i][1]))
        ry.append(list_2[i][0])

    rx_out = rx
    rx = np.array(rx)
    ry = np.array(ry)
    rx = rx[:, np.newaxis]

    def PolynomialRegression(degree=power1, **kwargs):
        return make_pipeline(PolynomialFeatures(degree),
                             LinearRegression(**kwargs))

    ypred = PolynomialRegression(degree=power1).fit(x, y).predict(x)

    def PolynomialRegression(degree=power2, **kwargs):
        return make_pipeline(PolynomialFeatures(degree),
                             LinearRegression(**kwargs))

    rypred = (PolynomialRegression(degree=power2).fit(rx, ry).predict(rx))

    return x_out, ypred, rx_out, rypred


def LVBound(VideoPath, ROI, LVPts, MV_LeftPts, MV_RightPts):
    """
    function:
        LVBound(VideoPath, ROI, LVPts, MV_LeftPts, MV_RightPts):

    parameters:
        VideoPath: AVI 檔案的輸入路徑, str.
        ROI: 超音波影像的有效區域
        LVPts: Left Ventricle 的中心點(Center), 經時間濾波(TimeMedian())後的結果.
        MV_LeftPts: Mitral Valve 左側支點, 經時間濾波(TimeMedian())後的結果.
        MV_RightPts: Mitral Valve 右側支點, 經時間濾波(TimeMedian())後的結果.

    return:
        lastL:
        lastR:
    """

    cap = cv2.VideoCapture(VideoPath)
    ROI = cv2.erode(ROI, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)  # ecg region
    frame_count = 0
    lastL, lastR = list(), list()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        text = "Frame_count: " + str(frame_count)
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (255, 255, 255), 1, cv2.LINE_AA)
        try:
            # Multi-Threshold
            multi = MultiThres(frame, ROI, 4, 40, 255)
            multi.SearchMax()
            frame_MT = multi.threshold()
            closing = cv2.morphologyEx(frame_MT, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                                       iterations=1)
            cnt, hierarchy = cv2.findContours(frame_MT, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnt:
                if cv2.contourArea(c) < 100:
                    cv2.drawContours(frame_MT, [c], 0, (0, 0, 0), -1)

            # 抓出每一幀的點 (左心室中心點) <- 經過時間濾波後的結果
            # 抓出每一幀的點 (二尖瓣左側支點, 二尖瓣右側支點) <- 經過時間濾波後的結果
            LVCenter = LVPts[frame_count - 1]
            MV_LeftPt = MV_LeftPts[frame_count - 1]
            MV_RightPt = MV_RightPts[frame_count - 1]

            # 掃描點(scanning) <- scan searching, 學長的模組
            pt = Rs.Rising_pt(LVCenter, frame_MT, 180, 8, 10, 5, 2)
            result = pt.search_pt()

            ScanInfo = {
                "Point": list(),
                "Distant": list(),
                "DistL": list(),  # 和二尖瓣左側支點的距離
                "DistR": list(),  # 和二尖瓣右側支點的距離
                "Apx": list()
            }

            for i in range(1, len(result) - 1, 1):
                y, x = int(result[i][0][0]), int(result[i][0][1])
                distant = 20 - int(result[i][1])
                dist_L = dist_pt(MV_LeftPt, (x, y))
                dist_R = dist_pt(MV_RightPt, (x, y))

                ScanInfo["Point"].append((x, y))
                ScanInfo["Distant"].append(distant)
                ScanInfo["Apx"].append(y)
                ScanInfo["DistL"].append(dist_L)
                ScanInfo["DistR"].append(dist_R)

                cv2.circle(closing, (x, y), 3, (192, 240, 150), -1)

            AP = ScanInfo["Apx"].index(min(ScanInfo["Apx"]))
            LI = ScanInfo["DistL"].index(min(ScanInfo["DistL"]))  # 找出與支點最近的掃描點後, 接著做線性回歸
            RI = ScanInfo["DistR"].index(min(ScanInfo["DistR"]))

            # 做線性回歸的左右側掃描點
            CavityL, CavityR = list(), list()
            for i in range(0, len(ScanInfo["Point"]), 1):
                x, y = ScanInfo["Point"][i]
                if AP >= i >= LI:
                    CavityL.append((x, y))

                if RI >= i >= AP:
                    CavityR.append((x, y))

            # 線性回歸
            reg = np.zeros([600, 800, 3], dtype=np.uint8)
            LRy_L, LRx_L, LRy_R, LRx_R = linear_LR(CavityL, CavityR, reg, LVCenter, 4, 3)
            LRPtx, LRPty = list(), list()

            # 順時針旋轉 LR 的點, 由於 LRx_L 和 LRx_R 點的數量不一定相同, 所以要拆開迴圈
            for i in range(len(LRx_L) - 1, -1, -1):
                LRPtx.append(int(LRx_L[i]))
            for i in range(len(LRx_R) - 1, -1, -1):
                LRPtx.append(int(LRx_R[i]))

            for i in range(len(LRy_L) - 1, -1, -1):
                LRPty.append(int(LRy_L[i]))
            for i in range(len(LRy_R) - 1, -1, -1):
                LRPty.append(int(LRy_R[i]))

            # check LR(檢查線性回歸的結果)
            checkLR = np.zeros([600, 800], dtype=np.uint8)
            for i in range(0, len(LRPtx), 1):
                if i == len(LRPtx) - 1:
                    cv2.line(checkLR, (LRPtx[i], LRPty[i]),
                             (LRPtx[0], LRPty[0]), (255, 255, 255), 1)
                else:
                    cv2.line(checkLR, (LRPtx[i], LRPty[i]),
                             (LRPtx[i + 1], LRPty[i + 1]), (255, 255, 255), 1)

            cnt, hierarchy = cv2.findContours(checkLR, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cnt, key=cv2.contourArea)
            cv2.drawContours(checkLR, [c], -1, (255, 255, 255), -1)
            # check end.

            # check Point(check result)
            checkPoint_frame = frame.copy()
            cv2.circle(checkPoint_frame, (LVCenter[0], LVCenter[1]), 3, (20, 140, 240), -1)

            LVScan = np.zeros([600, 800], dtype=np.uint8)
            for i in range(0, len(ScanInfo["Point"]), 1):
                if i == len(ScanInfo["Point"]) - 1:
                    cv2.line(LVScan, (ScanInfo["Point"][i][0], ScanInfo["Point"][i][1]),
                             (ScanInfo["Point"][0][0], ScanInfo["Point"][0][1]), (255, 255, 255), 1)
                else:
                    cv2.line(LVScan, (ScanInfo["Point"][i][0], ScanInfo["Point"][i][1]),
                             (ScanInfo["Point"][i + 1][0], ScanInfo["Point"][i + 1][1]), (255, 255, 255), 1)

            cnt, hierarchy = cv2.findContours(LVScan, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cnt, key=cv2.contourArea)
            cv2.drawContours(LVScan, [c], -1, (255, 255, 255), -1)
            # check end.

            # 找 LV Size
            pt = Rs.Rising_pt(LVCenter, frame_MT, 180, 8, 10, 5, 1)
            result = pt.search_pt()

            SizeInfo = {
                "Point": list(),
                "Distant": list(),
            }
            for i, pts in enumerate(result):
                if i > 0:
                    y, x = int(result[i][0][0]), int(result[i][0][1])
                    distant = 20 - int(result[i][1])

                    SizeInfo["Point"].append((x, y))
                    SizeInfo["Distant"].append(distant)

            # check LVSize (result)
            LVArea = np.zeros([600, 800], dtype=np.uint8)
            for i in range(1, len(SizeInfo["Point"]) - 1, 1):
                if i == len(SizeInfo["Point"]) - 1:
                    cv2.line(LVArea, (SizeInfo["Point"][i][0], SizeInfo["Point"][i][1]),
                             (SizeInfo["Point"][0][0], SizeInfo["Point"][0][1]), (255, 255, 255), 1)
                else:
                    cv2.line(LVArea, (SizeInfo["Point"][i][0], SizeInfo["Point"][i][1]),
                             (SizeInfo["Point"][i + 1][0], SizeInfo["Point"][i + 1][1]), (255, 255, 255), 1)
            # check end.

            # Mitral Valve function
            # 找 x, y 線段連線的係數 y = mx + b
            Xpt = [int(MV_LeftPt[0]), int(MV_RightPt[0])]
            Ypt = [int(MV_LeftPt[1]), int(MV_RightPt[1])]
            coefficients = np.polyfit(Xpt, Ypt, 1)
            m, b = coefficients[0], coefficients[1]
            y1, y2 = int(m + b), int(m * 800 + b)

            # 畫 LVArea 的 ConvexHull
            checkLVArea_frame = frame.copy()
            cnt, hierarchy = cv2.findContours(LVArea, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnt:
                hull = cv2.convexHull(c)

                cv2.drawContours(LVArea, [hull], 0, (255, 255, 255), -1)
                cv2.drawContours(checkLVArea_frame, [hull], 0, (255, 255, 255), 4)

                cv2.line(LVArea, (0, y1 + 100), (800, y2 + 100), (0, 0, 0), 100)

            LVArea_bgr = cv2.cvtColor(LVArea, cv2.COLOR_GRAY2BGR)
            test = cv2.bitwise_and(frame, LVArea_bgr)  # 將 LVArea 和 frame 的結果繪製在一起
            # -----

            # 疊加結果, 繪製出最大的 contour
            test_gr = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
            val, count = np.unique(test_gr, return_counts=True)
            avg = np.sum(val * count / np.sum(count[1:]))
            ret_S, thres = cv2.threshold(test_gr, avg * 1.5, 255, cv2.THRESH_BINARY_INV)

            thres = cv2.bitwise_and(thres, LVArea)
            thres = cv2.morphologyEx(thres, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                                     iterations=4)
            thres = cv2.erode(thres, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

            lst = cv2.bitwise_and(thres, LVArea)
            lst = cv2.erode(lst, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

            cnt, hierarchy = cv2.findContours(lst, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cnt, key=cv2.contourArea)

            LV_mor = np.zeros([600, 800], dtype=np.uint8)
            cv2.drawContours(LV_mor, [c], -1, (255, 255, 255), -1)
            # -----

            # 結合和線性回歸掃描點的結果
            checkLR = cv2.dilate(checkLR, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
            LV_mor = cv2.bitwise_and(LV_mor, checkLR)
            cnt, hierarchy = cv2.findContours(LV_mor, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cnt, key=cv2.contourArea)

            # 繪製最大的 contour 在 frame 上面
            DistL, DistR = list(), list()
            tempY = list()

            for i in range(0, len(c), 1):
                x, y = int(c[i][0][0]), int(c[i][0][1])
                distL = dist_pt(MV_LeftPt, (x, y))
                DistL.append(distL)

                distR = dist_pt(MV_RightPt, (x, y))
                DistR.append(distR)
                tempY.append(y)

            LI = DistL.index(min(DistL))
            RI = DistR.index(min(DistR))
            ap = tempY.index(min(tempY))

            CavityL, CavityR = list(), list()
            for i in range(0, len(c), 1):
                x, y = int(c[i][0][0]), int(c[i][0][1])
                if len(c) >= i >= RI:
                    CavityL.append((x, y))
                elif LI >= i >= ap:
                    CavityR.append((x, y))

            CavityL.sort(key=lambda s: s[1])
            CavityR.sort(key=lambda s: s[1])

            # 再一次線性回歸
            LRy_L, LRx_L, LRy_R, LRx_R = linear_LR(CavityL, CavityR, reg, LVCenter, 4, 3)
            ResPtL, ResPtR = list(), list()
            divL, divR = len(LRy_L) / 20, len(LRx_R) / 20

            for i in range(20, 0, -1):
                k = int(i * divR) - 1
                cv2.circle(frame, (int(LRx_R[k]), int(LRy_R[k])), 3, (240, 153, 216), -1)
                ResPtR.append((int(LRx_R[k]), int(LRy_R[k])))

                k = int(i * divL) - 1
                cv2.circle(frame, (int(LRx_L[k]), int(LRy_L[k])), 3, (240, 153, 216), -1)
                ResPtL.append((int(LRx_L[k]), int(LRy_L[k])))

            ResPtL = TimeMedian(ResPtL)
            ResPtR = TimeMedian(ResPtR)
            lastL.append(ResPtL)
            lastR.append(ResPtR)

        except:
            L_emt = [(0, 0)] * 20
            R_emt = [(0, 0)] * 20
            lastL.append(L_emt)
            lastR.append(R_emt)

    return lastL, lastR

