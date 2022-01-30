class vfs_params:
    def __init__(self):
        self.width = 848
        self.height = 800
        self.beta = 9.0
        self.gamma = 0.85
        self.alpha0 = 17.0
        self.alpha1 = 1.2
        self.timeStepLambda = 1.0
        self.Lambda = 5.0
        self.nLevel = 11
        self.fScale = 1.2
        self.nWarpIters = 100
        self.nSolverIters = 100
        self.limitRange = 0.1

class kb_params:
    def __init__(self):
        self.focalx = 285.722
        self.focaly = 286.759
        self.cx = 420.135
        self.cy = 403.394
        self.d0 = -0.00659769
        self.d1 = 0.0473251
        self.d2 = -0.0458264
        self.d3 = 0.00897725
        self.tx = -0.0641854
        self.ty = -0.000218299
        self.tz = 0.000111253
