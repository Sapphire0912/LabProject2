from MultiThreshold import *
from sklearn.cluster import KMeans
import FileIO
import numpy as np
import cv2
import os


class Segment(object):
    """
    class name:
        Segment(VideoPath, ROI, OutputSegDir)
        Apical four chamber view 的 腔室語意分析以及定義二尖瓣(Mitral Valve)位置

    parameters:
        VideoPath: 輸入影片的路徑, str
        ROI: 輸入要計算的 ROI 範圍 (二值圖)
        OutputSegDir: 輸入影片的路徑, str
    """
    def __init__(self, VideoPath, ROI, OutputSegDir):
        self.VideoPath = VideoPath
        self.roi = ROI[0]
        self.OutputSegDir = OutputSegDir

        # HandleHeartBound 屬性
        self.ChamberCenX, self.ChamberCenY = 0, 0
        self.MaskChamberBound = None

        # Semantic_FindValve 屬性
        self.Centroids = {"LV": (), "LA": (), "RV": (), "RA": ()}
        self.HistoryCenters = {"LV": [], "LA": [], "RV": [], "RA": []}

        self.LeftPivotList = list()
        self.RightPivotList = list()

    def HandleHeartBound(self, Skeleton):
        """
        method name:
            HandleHeartBound(Skeleton):
            定義超音波影像中心臟的範圍

        parameters:
            Skeleton: 輸入骨架圖(3通道), ndarray
        """
        print(f'----- 正在處理 {self.VideoPath} Segmentation -----')
        GraySkeleton = cv2.cvtColor(Skeleton, cv2.COLOR_BGR2GRAY)
        GraySkeleton[self.roi != 255] = 0

        try:
            area_ADD = cv2.morphologyEx(GraySkeleton, cv2.MORPH_CLOSE,
                                        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
            t = GraySkeleton.ravel()

            if t.max() > 100:
                cnt, _ = cv2.findContours(area_ADD, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                hull = list()

                for i in range(len(cnt)):
                    for j in range(len(cnt[i])):
                        hull.append(cnt[i][j])

                hull = np.asarray(hull)
                hull = cv2.convexHull(hull)

                bound = np.zeros((600, 800), np.uint8)
                car_bound = np.zeros((600, 800), np.uint8)

                cv2.drawContours(bound, [hull], 0, (255, 255, 255), -1)
                bound = cv2.erode(bound, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=9)
                cnt_B, _ = cv2.findContours(bound, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(car_bound, cnt_B, 0, (255, 255, 255), -1)
                cnt_car_bound, _ = cv2.findContours(car_bound, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                M = cv2.moments(cnt_car_bound[0])
                self.ChamberCenX, self.ChamberCenY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                self.MaskChamberBound = car_bound

        except ValueError:
            raise ValueError(f'當前骨架圖可能為全黑的')

    def _CreateFeatures(self):
        """
        method name:
            _CreateFeatures():
            創建給 Kmeans 訓練的特徵

        return:
            CenterData: Kmeans 的特徵矩陣, ndarray
        """
        video = cv2.VideoCapture(self.VideoPath)

        CenterData = list()
        mask_roi = self.roi
        ChamberCx, ChamberCy = self.ChamberCenX, self.ChamberCenY

        while True:
            ret, frame = video.read()

            if not ret:
                break

            frame[mask_roi != 255] = [0, 0, 0]
            median = cv2.medianBlur(frame, 19)

            gray_median = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
            gray_median[mask_roi != 255] = 0

            # - distance transform
            thres = cv2.adaptiveThreshold(gray_median, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 221, 7)
            thres[mask_roi != 255] = 0
            _, thres_inv = cv2.threshold(thres, 120, 255, cv2.THRESH_BINARY_INV)
            thres_inv[mask_roi != 255] = 0

            dist_cb = cv2.distanceTransform(thres_inv, cv2.DIST_L1, 3)
            dist_ms = cv2.distanceTransform(thres, cv2.DIST_L1, 3)

            cv2.normalize(dist_cb, dist_cb, 0, 1.0, cv2.NORM_MINMAX)
            cv2.normalize(dist_ms, dist_ms, 0, 1.0, cv2.NORM_MINMAX)

            _, dist = cv2.threshold(dist_cb, 0.6, 255, cv2.THRESH_BINARY)
            dist = np.uint8(dist)
            dist[self.MaskChamberBound != 255] = 0
            # - distance transform End.

            # -- 取出腔室中心點 & 創建特徵
            # - 取出腔室中心點
            cnt_center, _ = cv2.findContours(dist, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnt_center:
                area = cv2.contourArea(c)

                if area > 10:
                    HullC = cv2.convexHull(c)
                    x, y, w, h = cv2.boundingRect(HullC)
                    CenterX, CenterY = int(x + w / 2), int(y + h / 2)
                    # - 取出腔室中心點 End.

                    # - 創建特徵
                    diff_w, diff_h = CenterX - ChamberCx, CenterY - ChamberCy

                    # 用角度做 (為了要和 象限 及 角度一致, 因此要先判斷方向)
                    R = np.sqrt((CenterX - ChamberCx) ** 2 + (CenterY - ChamberCy) ** 2)
                    if R == 0:
                        theta = 0
                    else:
                        theta = np.degrees(np.arccos((CenterX - ChamberCx) / R))

                    # 由於 numpy & math 角度亂轉, 要多一點判斷式子
                    # 角度固定從 3 點鐘方向 順時針轉
                    if theta > 90:
                        theta -= 90
                    if abs(diff_h) > abs(diff_w):
                        theta = 90 - theta if theta < 45 else theta
                    if abs(diff_w) > abs(diff_h):
                        theta = 90 - theta if theta > 45 else theta

                    # LA 位置(+, +) 角度小於 90(不影響象限角)
                    if diff_w >= 0 and diff_h <= 0:  # LV 位置 (+, -) 角度 270 ~ 360
                        theta = 360 - theta
                    elif diff_w <= 0 and diff_h >= 0:  # RA 位置 (-, +) 角度 90 ~ 180
                        theta = 180 - theta
                    elif diff_w <= 0 and diff_h <= 0:  # RV 位置 (-, -) 角度 180 ~ 270
                        theta = 180 + theta

                    Feature = [diff_w, diff_h, theta, CenterX, CenterY]
                    CenterData.append(Feature)
                    # - 創建特徵 End.
            # -- 取出腔室中心點 & 創建特徵 End.

        return CenterData

    def _KmeansCluster(self, ClusterData):
        """
        method name:
            _KmeansCluster(ClusterData):
            利用 Kmeans 分群, 計算出 4 個腔室的質心位置

        parameters:
            ClusterData: 輸入 _CreateFeatures 的回傳值

        return:
            _KmeansAnomalyDetection(*params): 針對 Kmeans 的結果做異常檢測處理
        """
        if len(ClusterData) != 0:
            ClusterData = np.asarray(ClusterData)
            FeatureData, YData = ClusterData[:, :-1], ClusterData[:, -1]
            kmeans = KMeans(n_clusters=4, n_init=10)
            kmeans.fit(FeatureData, YData)
            YPred = kmeans.predict(FeatureData)

            # - 取出 Kmeans 的質心位置
            Cluster_Centers = kmeans.cluster_centers_

            Centroid_pts = list()
            for cluster_index, feature in enumerate(Cluster_Centers):
                dx, dy, angle, Xpoint = feature

                # 改用 dx, dy 去抓質心位置(沒有差太多, 而且相對來說角度較準)
                Xpt = int(self.ChamberCenX + dx)
                Ypt = int(self.ChamberCenY + dy)
                Centroid_pts.append((Xpt, Ypt))
            # - 取出 Kmeans 的質心位置 End.
        else:
            raise IndexError()

        return self._KmeansAnomalyDetection(
            ClusterFeature=Cluster_Centers,
            CentroidList=Centroid_pts,
            OriginalData=ClusterData,
            firstPred=YPred
        )

    def _KmeansAnomalyDetection(self, ClusterFeature, CentroidList, OriginalData, firstPred):
        """
        method name:
            _KmeansAnomalyDetection(ClusterFeature, CentroidList, OriginalData, firstPred):
            針對 Kmeans 的結果做異常檢測處理

        parameters:
            參數皆為 _KmeansCluster() 的回傳值

        return:
            adjust_Centroid: 調整過後的 Kmeans 4 個腔室的質心位置
        """
        adjust_Centroid = {"LV": (), "LA": (), "RV": (), "RA": ()}

        # --- 根據 4 個質心位置 繪製四邊形找出四邊形中心
        CentroidList = np.asarray(CentroidList)
        quadrilateral = cv2.convexHull(CentroidList)
        quadM = cv2.moments(quadrilateral)
        quad_x, quad_y = int(quadM["m10"] / quadM["m00"]), int(quadM["m01"] / quadM["m00"])
        # --- 根據 4 個質心位置 繪製四邊形找出四邊形中心 End.

        # --- 區分以四邊形為中心點的左右資料
        left, left_index = len(np.where(CentroidList[:, 0] < quad_x)[0]), np.where(CentroidList[:, 0] < quad_x)[0]
        right, right_index = len(np.where(CentroidList[:, 0] >= quad_x)[0]), np.where(CentroidList[:, 0] >= quad_x)[0]
        # --- 區分以四邊形為中心點的左右資料 End.

        if left != right:
            if right > left:
                LeftData = OriginalData[firstPred == left_index]
                RightData = OriginalData[firstPred != left_index]

            else:
                LeftData = OriginalData[firstPred != right_index]
                RightData = OriginalData[firstPred == right_index]

            LeftCentroid, LeftPred = self._ReKmeans(LeftData)
            RightCentroid, RightPred = self._ReKmeans(RightData)

            # -- 判斷是否分錯(緩)
            adjust_Centroid["RA"] = LeftCentroid[0] if LeftCentroid[0][1] > LeftCentroid[1][1] else LeftCentroid[1]
            adjust_Centroid["RV"] = LeftCentroid[1] if LeftCentroid[0][1] > LeftCentroid[1][1] else LeftCentroid[0]

            adjust_Centroid["LA"] = RightCentroid[0] if RightCentroid[0][1] > RightCentroid[1][1] else RightCentroid[1]
            adjust_Centroid["LV"] = RightCentroid[1] if RightCentroid[0][1] > RightCentroid[1][1] else RightCentroid[0]

        else:
            # 若位置正確, 按象限角分沒問題
            thetaData = ClusterFeature[:, 2]
            thetaSort = np.argsort(thetaData)

            adjust_Centroid["LA"] = tuple(CentroidList[thetaSort[0]])
            adjust_Centroid["RA"] = tuple(CentroidList[thetaSort[1]])
            adjust_Centroid["RV"] = tuple(CentroidList[thetaSort[2]])
            adjust_Centroid["LV"] = tuple(CentroidList[thetaSort[3]])

        return adjust_Centroid

    def _ReKmeans(self, data):
        xdata, ydata = data[:, :-1], data[:, -1]
        kmeans2 = KMeans(n_clusters=2, n_init=6)
        kmeans2.fit(xdata, ydata)
        ypred = kmeans2.predict(xdata)
        # --- 利用 Kmeans 區分四個腔室 End.

        # --- 定義質心位置
        Cluster_Centers2 = kmeans2.cluster_centers_

        Centroid_pts2 = list()
        for _feature in Cluster_Centers2:
            _dx, _dy, _angle, _ = _feature
            xpt = int(self.ChamberCenX + _dx)
            ypt = int(self.ChamberCenY + _dy)
            Centroid_pts2.append((xpt, ypt))

        # --- 定義質心位置
        return Centroid_pts2, ypred

    def _FrameCenterAnomalyDetection(self, CurrentCenters):
        """
        method name:
            _FrameCenterAnomalyDetection(CurrentCenters):
            處理每幀的腔室中心點位置不正確的情況

        parameters:
            CurrentCenters: 輸入該幀的腔室中心點, dict

        return:
            CurrentCenters: 回傳經異常處理後的 4 個腔室位置, dict
        """
        # -- 處理腔室多個點的情況
        not_exist = list()
        for pos in CurrentCenters.keys():
            if len(CurrentCenters[pos]) > 1:
                if len(CurrentCenters[pos]) == 2:
                    x1, y1 = CurrentCenters[pos][0]
                    x2, y2 = CurrentCenters[pos][1]
                    CurrentCenters[pos] = ((x2 - x1) // 2 + x1, (y2 - y1) // 2 + y1)
                    self.HistoryCenters[pos].append(((x2 - x1) // 2 + x1, (y2 - y1) // 2 + y1))

                else:
                    L = np.asarray(CurrentCenters[pos])
                    Hull = cv2.convexHull(L)
                    Hull_M = cv2.moments(Hull)
                    if Hull_M["m00"] != 0:
                        Cx, Cy = int(Hull_M["m10"] / Hull_M["m00"]), int(Hull_M["m01"] / Hull_M["m00"])
                        CurrentCenters[pos] = (Cx, Cy)
                        self.HistoryCenters[pos].append((Cx, Cy))

                    else:
                        not_exist.append(pos)

            if len(CurrentCenters[pos]) == 1:
                CurrentCenters[pos] = CurrentCenters[pos][0]
                self.HistoryCenters[pos].append(tuple(CurrentCenters[pos]))

            if len(CurrentCenters[pos]) == 0:
                not_exist.append(pos)
        # -- 處理腔室多個點的情況  End.

        # -- 處理當前腔室不存在的問題
        # 利用歷史的中心點, 取正中心點 取代(預測)當前幀的中心
        for nan_pos in not_exist:
            history_pos = self.HistoryCenters[nan_pos]
            # 避免因為每次都是加入質心點或歷史座標, 而出現 convexHull 的 bug(即使長度 > 2, 但實質上只有一點, 無法成線和凸包)
            # 去除重複的部分, 再進行判斷; 但不影響 HistoryCenters 紀錄
            history_pos = set(history_pos)
            history_pos = list(history_pos)

            if len(history_pos) != 0:
                if len(history_pos) == 1:
                    CurrentCenters[nan_pos] = self.HistoryCenters[nan_pos][0]
                    self.HistoryCenters[nan_pos].append(self.HistoryCenters[nan_pos][0])

                if len(history_pos) == 2:
                    x1, y1 = self.HistoryCenters[nan_pos][0]
                    x2, y2 = self.HistoryCenters[nan_pos][1]
                    CurrentCenters[nan_pos] = ((x2 - x1) // 2 + x1, (y2 - y1) // 2 + y1)
                    self.HistoryCenters[nan_pos].append(((x2 - x1) // 2 + x1, (y2 - y1) // 2 + y1))

                if len(history_pos) > 2:
                    L = np.asarray(self.HistoryCenters[nan_pos])
                    Hull = cv2.convexHull(L)
                    Hull_M = cv2.moments(Hull)
                    if Hull_M["m00"] != 0:
                        Cx, Cy = int(Hull_M["m10"] / Hull_M["m00"]), int(Hull_M["m01"] / Hull_M["m00"])
                    else:
                        Cx, Cy = self.Centroids[nan_pos]

                    CurrentCenters[nan_pos] = (Cx, Cy)
                    self.HistoryCenters[nan_pos].append((Cx, Cy))

            else:
                # 若無歷史紀錄, 則用質心點取代
                CurrentCenters[nan_pos] = self.Centroids[nan_pos]
                self.HistoryCenters[nan_pos].append(self.Centroids[nan_pos])
        # -- 處理當前腔室不存在的問題 End.

        # -- 處理腔室中心點過於中心區域及邊界的問題(這部分應該要對所有腔室做)
        for near_pos in CurrentCenters.keys():
            LVx, LVy = CurrentCenters["LV"]
            LAx, LAy = CurrentCenters["LA"]
            RVx, RVy = CurrentCenters["RV"]
            RAx, RAy = CurrentCenters["RA"]

            CentroidLVx, CentroidLVy = self.Centroids["LV"]
            CentroidLAx, CentroidLAy = self.Centroids["LA"]
            CentroidRVx, CentroidRVy = self.Centroids["RV"]
            CentroidRAx, CentroidRAy = self.Centroids["RA"]

            LeftYDiff = CentroidLAy - CentroidLVy
            RightYDiff = CentroidRAy - CentroidRVy

            UpXDiff = CentroidLVx - CentroidRVx
            DownXDiff = CentroidLAx - CentroidRAx

            if near_pos == "LA":
                up_limit = int(LeftYDiff * 0.35) + CentroidLVy
                down_limit = int(LeftYDiff * 0.65) + CentroidLVy
                left_limit = int(DownXDiff * 0.35) + CentroidRAx
                right_limit = int(DownXDiff * 0.65) + CentroidRAx

                isCenter = (up_limit <= LAy <= down_limit) or (left_limit <= LAx <= right_limit)

                if isCenter:
                    CurrentCenters["LA"] = (CentroidLAx, CentroidLAy)
                    self.HistoryCenters["LA"][-1] = (CentroidLAx, CentroidLAy)

            elif near_pos == "LV":
                up_limit = int(LeftYDiff) * 0.35 + CentroidLVy
                down_limit = int(LeftYDiff) * 0.65 + CentroidLVy
                left_limit = int(UpXDiff * 0.35) + CentroidRVx
                right_limit = int(UpXDiff * 0.65) + CentroidRVx

                isCenter = (up_limit <= LVy <= down_limit) or (left_limit <= LVx <= right_limit)

                if isCenter:
                    CurrentCenters["LV"] = (CentroidLVx, CentroidLVy)
                    self.HistoryCenters["LV"][-1] = (CentroidLVx, CentroidLVy)

            elif near_pos == "RV":
                up_limit = int(RightYDiff) * 0.35 + CentroidRVy
                down_limit = int(RightYDiff) * 0.65 + CentroidRVy
                left_limit = int(UpXDiff * 0.35) + CentroidRVx
                right_limit = int(UpXDiff * 0.65) + CentroidRVx

                isCenter = (up_limit <= RVy <= down_limit) or (left_limit <= RVx <= right_limit)

                if isCenter:
                    CurrentCenters["RV"] = (CentroidRVx, CentroidRVy)
                    self.HistoryCenters["RV"][-1] = (CentroidRVx, CentroidRVy)

            elif near_pos == "RA":
                up_limit = int(RightYDiff) * 0.35 + CentroidRVy
                down_limit = int(RightYDiff) * 0.65 + CentroidRVy
                left_limit = int(DownXDiff * 0.35) + CentroidRAx
                right_limit = int(DownXDiff * 0.65) + CentroidRAx

                isCenter = (up_limit <= RAy <= down_limit) or (left_limit <= RAx <= right_limit)

                if isCenter:
                    CurrentCenters["RA"] = (CentroidRAx, CentroidRAy)
                    self.HistoryCenters["RA"][-1] = (CentroidRAx, CentroidRAy)

        # - 處理腔室中心點過於中心的問題 End.
        return CurrentCenters

    def Semantic_FindValve(self, isOutputSegVideo=False):
        """
        method name:
            Semantic_FindValve(isOutputSegVideo=False):
            Semantic Apical four chamber view 的腔室位置, 以及二尖瓣(Mitral Valve)位置

        parameters:
            isOutputSegVideo: 是否輸入 Semantic 的影片, 默認 False
        """
        CenterData = self._CreateFeatures()
        self.Centroids = self._KmeansCluster(CenterData)

        Centroid = self.Centroids
        mask_roi = self.roi

        if not os.path.isdir(self.OutputSegDir):
            os.makedirs(self.OutputSegDir)

        video = cv2.VideoCapture(self.VideoPath)
        ChambersColors = [(255, 0, 255), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # LV, LA, RV, RA
        semantic_list = list()
        frame_count = 0
        TextFont = cv2.FONT_HERSHEY_COMPLEX_SMALL

        while True:
            ret, frame = video.read()

            if not ret:
                break

            frame_count += 1
            frame[mask_roi != 255] = [0, 0, 0]
            DrawSemantic = frame.copy()
            FrameValve = frame.copy()

            median = cv2.medianBlur(frame, 19)
            gray_median = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
            gray_median[mask_roi != 255] = 0

            # distance transform
            thres = cv2.adaptiveThreshold(gray_median, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 221, 7)
            thres[mask_roi != 255] = 0
            _, thres_inv = cv2.threshold(thres, 120, 255, cv2.THRESH_BINARY_INV)
            thres_inv[mask_roi != 255] = 0

            dist_cb = cv2.distanceTransform(thres_inv, cv2.DIST_L1, 3)
            dist_ms = cv2.distanceTransform(thres, cv2.DIST_L1, 3)

            cv2.normalize(dist_cb, dist_cb, 0, 1.0, cv2.NORM_MINMAX)
            cv2.normalize(dist_ms, dist_ms, 0, 1.0, cv2.NORM_MINMAX)
            _, dist = cv2.threshold(dist_cb, 0.6, 255, cv2.THRESH_BINARY)
            dist = np.uint8(dist)
            dist[self.MaskChamberBound != 255] = 0

            cnt_center, _ = cv2.findContours(dist, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            curr_center = {"LV": [], "LA": [], "RV": [], "RA": []}

            for c in cnt_center:
                area = cv2.contourArea(c)

                if area > 10:
                    HullC = cv2.convexHull(c)
                    x, y, w, h = cv2.boundingRect(HullC)
                    CenterX, CenterY = int(x + w / 2), int(y + h / 2)
                    # - 到這裡步驟和前面一致 End.

                    # -- 利用距離判斷當前 Center 和 Centroid 的關係
                    DisLV = np.sqrt((CenterX - Centroid["LV"][0]) ** 2 + (CenterY - Centroid["LV"][1]) ** 2)
                    DisLA = np.sqrt((CenterX - Centroid["LA"][0]) ** 2 + (CenterY - Centroid["LA"][1]) ** 2)
                    DisRV = np.sqrt((CenterX - Centroid["RV"][0]) ** 2 + (CenterY - Centroid["RV"][1]) ** 2)
                    DisRA = np.sqrt((CenterX - Centroid["RA"][0]) ** 2 + (CenterY - Centroid["RA"][1]) ** 2)
                    DisList = [DisLV, DisLA, DisRV, DisRA]
                    MinDis = DisList.index(min(DisList))

                    if MinDis == 0:
                        curr_center["LV"].append((CenterX, CenterY))
                    elif MinDis == 1:
                        curr_center["LA"].append((CenterX, CenterY))
                    elif MinDis == 2:
                        curr_center["RV"].append((CenterX, CenterY))
                    elif MinDis == 3:
                        curr_center["RA"].append((CenterX, CenterY))
                    # -- 利用距離判斷當前 Center 和 Centroid 的關係 End.

            # -- 針對每幀腔室的點做異常檢測
            curr_center = self._FrameCenterAnomalyDetection(
                CurrentCenters=curr_center
            )
            cv2.putText(DrawSemantic, 'LV', curr_center["LV"], TextFont, 1, ChambersColors[0], 1)
            cv2.putText(DrawSemantic, 'LA', curr_center["LA"], TextFont, 1, ChambersColors[1], 1)
            cv2.putText(DrawSemantic, 'RV', curr_center["RV"], TextFont, 1, ChambersColors[2], 1)
            cv2.putText(DrawSemantic, 'RA', curr_center["RA"], TextFont, 1, ChambersColors[3], 1)
            # cv2.putText(DrawSemantic, f'frame count: {frame_count}', (100, 100), TextFont, 1, (255, 255, 255), 1)

            # Handle Valve
            # --- 根據每幀抓取瓣膜位置
            # 利用 multi-threshold
            frame_valve_gray = cv2.cvtColor(FrameValve, cv2.COLOR_BGR2GRAY)
            multi = MultiThres(frame_valve_gray, self.roi, 3, 40, 255)
            multi.SearchMax()
            Multi = multi.threshold()
            Multi_BGR = cv2.cvtColor(Multi, cv2.COLOR_GRAY2BGR)

            mask_region = np.zeros((600, 800), np.uint8)
            cnt_region = np.asarray(
                [curr_center["LV"], curr_center["LA"],
                 curr_center["RA"], curr_center["RV"]]
            )
            cv2.drawContours(mask_region, [cnt_region], -1, (255, 255, 255), -1)
            mask_region = cv2.erode(mask_region, np.ones((3, 3)), iterations=15)

            Multi[mask_region != 255] = 0

            cnt_Multi, _ = cv2.findContours(Multi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_filter = np.zeros((600, 800), np.uint8)
            for cnt_index, cnt in enumerate(cnt_Multi):
                if cv2.contourArea(cnt) > 100:
                    cv2.drawContours(mask_filter, [cnt], -1, (255, 255, 255), -1)
                    cv2.drawContours(Multi_BGR, [cnt], -1, (0, 0, 255), -1)

            cnt_filter, _ = cv2.findContours(mask_filter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            HullFilter = list()

            for i in range(len(cnt_filter)):
                for j in range(len(cnt_filter[i])):
                    HullFilter.append(cnt_filter[i][j])

            if len(HullFilter) != 0:
                HullFilter = np.asarray(HullFilter)
                HullFilter = cv2.convexHull(HullFilter)
                MFilter = cv2.moments(HullFilter)
                LeftPivotX, LeftPivotY = int(MFilter["m10"] / MFilter["m00"]), int(MFilter["m01"] / MFilter["m00"])

                # cv2.line(DrawSemantic, curr_center["LV"], curr_center["LA"], (255, 0, 0), 2)  # 視覺化用(LV & LA 線段)

                # 計算左支點到腔室中心連線段的距離
                CenLVX, CenLVY = curr_center["LV"]
                CenLAX, CenLAY = curr_center["LA"]
                if CenLAX != CenLVX and CenLVY != CenLVY:
                    slope_LVLA = (CenLAY - CenLVY) / (CenLAX - CenLVX)
                    PivotDis = int((LeftPivotY - CenLVY + slope_LVLA * CenLVX) / slope_LVLA) - LeftPivotX + 2
                else:
                    PivotDis = int(CenLVX - LeftPivotX) + 2
                RightPivotX = LeftPivotX + 2 * PivotDis

                self.LeftPivotList.append((LeftPivotX, LeftPivotY))
                self.RightPivotList.append((RightPivotX, LeftPivotY))
                # cv2.drawContours(DrawSemantic, [HullFilter], -1, (0, 255, 0), 2)

            else:
                LeftPivotX, LeftPivotY = self.LeftPivotList[-1]
                RightPivotX = self.RightPivotList[-1][0]
                self.LeftPivotList.append(self.LeftPivotList[-1])
                self.RightPivotList.append(self.RightPivotList[-1])

            cv2.line(DrawSemantic, (LeftPivotX, LeftPivotY), (RightPivotX, LeftPivotY), (255, 255, 0), 2)
            cv2.circle(DrawSemantic, (LeftPivotX, LeftPivotY), 10, (0, 255, 255), -1)
            cv2.circle(DrawSemantic, (RightPivotX, LeftPivotY), 10, (0, 255, 255), -1)
            semantic_list.append(DrawSemantic)

        if isOutputSegVideo:
            FileName = self.VideoPath.split('\\')[-1]
            FileIO.write_video(semantic_list, self.OutputSegDir + ' semantic_valve ' + FileName)
