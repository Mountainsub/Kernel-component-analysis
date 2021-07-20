from sklearn.preprocessing import KernelCenterer

class KernelPca:
    # beta: ガウスカーネルパラメータ
    def __init__(self, beta):
        self.beta = beta
        self.centerer = KernelCenterer()

    # gauss kernel
    def __kernel(self, x1, x2):
        return np.exp(-self.beta * np.linalg.norm(x1 - x2)**2)

    # データを入力して主成分ベクトルを計算する
    # shape(X) = (N, M)
    # n: 抽出する主成分の数
    def fit_transform(self, X, n):
        self.X = X
        # グラム行列
        N = X.shape[0]
        K = np.array(
            [[self.__kernel(X[i], X[j]) for j in range(N)] for i in range(N)])
        # 中心化
        K = self.centerer.fit_transform(K)
        # eighは固有値の昇順で出力される
        vals, vecs = np.linalg.eigh(K)
        vals = vals[::-1]
        vecs = vecs[:, ::-1]
        # 特異値と左特異ベクトル、上位n個
        self.sigma = np.sqrt(vals[:n])  # (n)
        self.a = np.array(vecs[:, :n])  # (N,n)
        return self.sigma * self.a      # (N,n)

    # xの主成分表示を返す
    # shape(x)=(Nx, M)
    def transform(self, x):
        # グラム行列
        N = self.X.shape[0]
        Nx = x.shape[0]
        K = np.array(
            [[self.__kernel(x[i], self.X[j]) for j in range(N)] for i in range(Nx)]
        )  # (Nx,N)
        # 中心化
        K = self.centerer.transform(K)
        # 主成分を計算
        return K.dot(self.a) / self.sigma  # (Nx,n)
