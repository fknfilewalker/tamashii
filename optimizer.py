from tamashii._spy import spy
import numpy as np

class SAdam(spy.InstanceTensor):
    def __init__(self, device: spy.Device, shape: tuple[int, ...], beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(struct=spy.Module.load_from_file(device, "tamashii.slang").find_struct("optimization::Adam<float>").as_struct(), shape=shape)
        self.construct(beta1=beta1, beta2=beta2, epsilon=epsilon)

    def update(self, lr: float, grad: spy.Tensor, param: spy.Tensor):
        self.__getattr__("update")(lr, grad, param)

class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.params = params.copy()
        self.m = np.zeros_like(self.params)
        self.v = np.zeros_like(self.params)
        self.t = 0

    def step(self, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        self.params -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)