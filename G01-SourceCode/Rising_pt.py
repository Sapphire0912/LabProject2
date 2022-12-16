import numpy as np
import matplotlib.pyplot as plt
# for p in range(-3, 4, 1):
#     x_p = 4 - abs(p)
#     for q in range(-(x_p - 1), x_p, 1):
#         print(p, q)
#         x = 2
#         y = 4
class Rising_pt:
    def __init__(self, C, I, R, A, S, sR, M):
        """
        C:  Center
        I:  Image
        R:  Radius
        A:  Angle
        S:  Standard Length
        sR: Score Radius
        M:  Mode
        """
        self.x      = C[1]
        self.y      = C[0]
        self.I = I
        self.IW= I.shape[1]
        self.IL= I.shape[0]
        self.R  = R
        self.A  = A
        self.S  = S
        self.sR = sR
        self.M  = M


    def search_pt(self):
        pt_lst = []
        div = int(360 / self.A)
        for i in range(0, div, 1):
            angle = self.A*i

            radians = np.radians(angle)
            sin = np.sin(radians)
            cos = np.cos(radians)


            div = self.R / self.S

            dis = []
            all_sc = []
            pt_org = []
            clockwise_pt_x = []
            clockwise_pt_y = []
            # print(angle,"......angle")
            # print(sin,cos)
            for k in range(1, int(div)+1, 1):
                pt_x  = int(self.x + (self.S * k) * cos)
                pt_y  = int(self.y - (self.S * k) * sin)
                area_pixel = []
                # pt_lst.append((pt_x,pt_y))

                for p in range(-(self.sR), self.sR+1, 1):
                    x_p = self.sR-abs(p)
                    for q in range(-(x_p - 1), x_p, 1):

                        try:
                            area_pixel.append(self.I[pt_x + q][pt_y + p])
                            sucess = True
                        except:
                            sucess = False
                            pass

                score = sum(area_pixel)
                all_sc.append(score)
                dis.append(k)
                pt_org.append((pt_x,pt_y))


                clockwise_pt_x.append(pt_x)
                clockwise_pt_y.append(pt_y)

            med_pt_x = []
            med_pt_y = []

            for i in range(0, len(clockwise_pt_x)-1, 1):


                if i+1 == len(clockwise_pt_x)-1:
                    st_x, st_y = clockwise_pt_x[i], clockwise_pt_y[i]
                    nd_x, nd_y = clockwise_pt_x[i + 1], clockwise_pt_y[i + 1]
                    rd_x, rd_y = nd_x, nd_y
                elif i == len(clockwise_pt_x)-1:
                    st_x, st_y = clockwise_pt_x[i], clockwise_pt_y[i]
                    nd_x, nd_y = st_x, st_y
                    rd_x, rd_y = nd_x, nd_y
                else:
                    st_x, st_y = clockwise_pt_x[i], clockwise_pt_y[i]
                    nd_x, nd_y = clockwise_pt_x[i + 1], clockwise_pt_y[i + 1]
                    rd_x, rd_y = clockwise_pt_x[i + 2], clockwise_pt_y[i + 2]

                if st_x <= nd_x <= rd_x or rd_x <= nd_x <= st_x:
                    med_pt_x.append(nd_x)
                elif nd_x <= st_x <= rd_x or rd_x <= st_x <= nd_x:
                    med_pt_x.append(st_x)
                else:
                    med_pt_x.append(rd_x)

                if st_y <= nd_y <= rd_y or rd_y <= nd_y <= st_y:
                    med_pt_y.append(nd_y)
                elif nd_y <= st_y <= rd_y or rd_y <= st_y <= nd_y:
                    med_pt_y.append(st_y)
                else:
                    med_pt_y.append(rd_y)
            pt_temp = []
            for i in range(0, len(med_pt_x) - 1, 1):
                pt_temp.append((med_pt_x[i], med_pt_y[i]))


            if self.M == 1:
                for i in range(0,len(pt_temp)-1,1):
                    sum_slp = sum(all_sc[:i])
                    if sum_slp > 5000 :
                        pt_lst.append((pt_temp[i-1],dis[i-1],angle))
                        break
                    else:
                        pass
            if self.M == 2:
                for i in range(0,len(pt_temp)-1,1):
                    sum_slp = sum(all_sc[:i])-sum(all_sc[:i-1])
                    if sum_slp > 200 :
                        if (i-2) > 0:
                            pt_lst.append((pt_temp[i-2],dis[i-2],angle))
                        else:
                            pt_lst.append((pt_temp[0], dis[0], angle))
                        # print(i,i-2)
                        break
                    else:
                        pass
            if self.M == 3:

                i = all_sc.index(max(all_sc))
                pt_lst.append((pt_org[i-1],dis[i-1],angle))

            # if self.M == 4:
            #     for i in range(0,len(pt_temp)-1,1):
            #         sum_slp = sum(all_sc[:i])-sum(all_sc[:i-1])
            #         # print(sum(all_sc[:i]),sum(all_sc[:i]),sum_slp)
            #         if sum_slp > 200 :
            #             pt_lst.append((pt_temp[i-1],dis[i-1],angle))
            #             break
            #         else:
            #             pass

            hor = []
            for i in range(1, len(pt_lst) - 1, 1):
                distant = 20 - int(pt_lst[i][1])
                hor.append(distant)

            # error_pt = []
            # for i in range(1, len(hor)-1, 1):
            #     if abs(hor[i]-hor[i+1]) > 3 and abs(hor[i]-hor[i-1]) > 3:
            #         error_pt.append(i)
            #
            # error_pt.reverse()
            # for i in error_pt:
            #     pt_lst.pop(i)


        return pt_lst