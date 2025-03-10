import torch.nn as nn
import numpy as np
from abc import abstractmethod

class BaseModel(nn.Module):
    """
    Base class for all models

    """


    @abstractmethod
    def forward(self, *inputs):
        """
        *input表示可以接受任何数量的参数
        抛出异常，强制子类执行
        前向传播逻辑
        :return: Model output
        """
        raise NotImplementedError


    def __str__(self):
        """
        __str__：当调用print打印时，自动调用该方法并输出返回值
        Model prints with number of trainable parameters

        self.parameters会返回所有可学习的参数，filter
        :return:
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
