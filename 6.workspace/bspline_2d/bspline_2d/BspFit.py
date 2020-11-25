from scipy.interpolate import BSpline, splev
import numpy as np

class BspFit:
    def __init__(self, dets, c_num=None, alpha=0):
        """
        bbx_dets: np.array. shape: #images * 4      (x1,y1,x2,y2)
        """
        self._dets = dets
        self._num = len(self._dets)
        self._k = 3                                                      # 3次B样条
        if c_num is None:
            self._c_num = self._num // 5 if self._num // 5 >=4 else 4   # 至少取4个控制点
        else:
            self._c_num = c_num
        self._alpha = alpha                                               # 正则项系数，逼近项系数为1

    def _get_PointList(self):
        # 待拟合的点列
        # 默认取检测框的底部中点
        self._P = [np.array(((box[0]+box[2])/2, box[3])) for box in self._dets]

    def _get_param(self):
        # 将待拟合的点参数化
        # 默认为弦长参数化
        dis = [np.linalg.norm(self._P[i] - self._P[i-1] ) for i in range(1,len(self._P))]
        dis.insert(0,0.0)
        dis = np.array(dis)
        dis = np.cumsum(dis)
        self._X = dis / dis[-1]
    
    def _get_knots(self):
        # 生成节点向量，均匀选择
        # 节点向量的数量可以指定，不指定时选择为样本量的 1/5
        # 更准确的说是指定控制点的数量，这里采用三次样条，节点向量长度为控制点数量 +4
        
        assert self._c_num >= 4 # 至少应该有4个控制点
        self._t = np.array([0.0]*3 + list(np.linspace(start=0,stop=1,num=self._c_num-2,endpoint=True)) + [1.0]*3)
    
    def _get_M(self):
        # 逼近项的系数矩阵
        M = np.zeros((len(self._P),self._c_num))
        for j in range(self._c_num):
            b = BSpline.basis_element(self._t[j:j+self._k+2])
            for i in range(len(self._P)):
                if self._t[j] <= self._X[i] < self._t[j+self._k+1]:
                    M[i,j] = b(self._X[i])
        
        self._M = M
    
    def _get_W(self):
        # 正则项矩阵
        
        # 辅助矩阵N  存储每一个基函数在一个小区间内二次导数的斜率和截距      
        N = np.zeros((self._c_num, self._c_num+self._k, 2), dtype=np.float)
        for i in range(self._c_num):
            t = self._t[i:i+self._k+2]
            spl = BSpline.basis_element(t)
            for j in range(4):
                if t[j] < t[j+1]:
                    a = (t[j+1] - t[j]) / 3 + t[j]
                    b = (t[j+1] - t[j]) / 3 + a

                    a_ = splev(a,spl,der=2)
                    b_ = splev(b,spl,der=2)

                    N[i,i+j,0] = (b_ - a_) / (b -a)
                    N[i,i+j,1] = (a*b_ -a_*b) / (a - b)
        
        # 填充矩阵W
        # 此处应该优化self_t
        def Inte(f1,f2,x1,x2):
            # 计算积分的辅助函数
            if x1 >= x2:
                return 0.0
            else:
                a,b = f1
                c,d = f2
                return -1 / 6 * (x1 - x2) * (3 * b * (2*d + c*(x1+x2)) + a*(3*d*(x1+x2) + 2*c*(x1**2 + x1*x2 +x2**2)))  

        W = np.zeros((self._c_num, self._c_num))
        for j in range(len(self._t)-1):
            x1, x2 = self._t[j:j+2]
            for row in range(self._c_num):
                for col in range(self._c_num):
                    W[row,col] += Inte((N[row,j,0],N[row,j,1]),(N[col,j,0],N[col,j,1]),x1,x2)
        self._W = W
    
    def _solve_init(self):
        assert self._alpha >= 0
        self._get_PointList()
        self._get_param()
        self._get_knots()
        self._get_M()

    # 对外接口

    def solve(self,alpha=None):
        self._solve_init()
        self._alpha = alpha if alpha is not None else self._alpha

        if self._alpha > 0:
            self._get_W()

            A = self._alpha * self._W + self._M.T.dot(self._M)
            b = self._M.T.dot(np.array(self._P))

            C = np.linalg.solve(A,b)
            tck = (self._t, C, self._k)
        else:
            # C = np.linalg.lstsq(self._M,np.array(self._P))[0]
            A = self._M.T.dot(self._M)
            b = self._M.T.dot(np.array(self._P))
            C = np.linalg.solve(A, b)
            tck = (self._t, C, self._k)

        return tck

    def set_alpha(self,alpha):
        self._alpha = alpha
    def set_c_num(self,n):
        self._c_num = n
    def set_get_PointList(self,method):
        pass
