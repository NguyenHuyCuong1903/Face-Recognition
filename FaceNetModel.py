import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class FaceNetModel(nn.Module):
    def __init__(self):
        super(FaceNetModel, self).__init__()
        self.facenet = InceptionResnetV1(pretrained='vggface2', classify=False)
        # for name, param in self.facenet.named_parameters():
        #     if name == "last_linear.weight" or name == "last_linear.bias" or name == "last_bn.weight" or name == "last_bn.bias":
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

    def forward(self, x):
        x = self.facenet(x)
        return x