
import torchvision
import torch.nn as nn
import torch
import timm
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import PyTorch_CIFAR10.cifar10_models.vgg as cifar10_vgg
import PyTorch_CIFAR10.cifar10_models.resnet as cifar10_resnet
import PyTorch_CIFAR10.cifar10_models.mobilenetv2 as cifar10_mobilenetv2
import PyTorch_CIFAR10.cifar10_models.inception as cifar10_inception
import PyTorch_CIFAR10.cifar10_models.densenet as cifar10_densenet
import PyTorch_CIFAR10.cifar10_models.googlenet as cifar10_googlenet
import DVERGE.utils
import DVERGE.eval.eval_bbox
from MadryLab import model_utils as Madry_model_utils
from MadryLab.datasets import ImageNet as Madry_ImageNet
import os
import torch.nn.functional as F
dir_path = os.path.dirname(os.path.realpath(__file__))

## simple wrapper model to normalize an input image
class WrapperModel(nn.Module):
    def __init__(self, model, mean, std,resize=False):
        super(WrapperModel, self).__init__()
        self.mean = torch.Tensor(mean)
        self.model=model
        self.resize=resize
        self.std = torch.Tensor(std)
    def forward(self, x):
        if self.resize==True:
            x = transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST)(x)
        return self.model((x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None])
    def normalize(self, x):
        if self.resize==True:
            x = transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST)(x)
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]



def load_model(model_name):
    if model_name == "ResNet101":
        model = torchvision.models.resnet101(pretrained=True)
    elif model_name == 'ResNet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif model_name == 'ResNet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif model_name == 'ResNet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif model_name == "ResNet152":
        model = torchvision.models.resnet152(pretrained=True)
    elif model_name == "vgg16":
        model = torchvision.models.vgg16_bn(pretrained=True)
    elif model_name == "vgg19":
        model = torchvision.models.vgg19_bn(pretrained=True)
    elif model_name == "wide_resnet101_2":
        model = torchvision.models.wide_resnet101_2(pretrained=True)
    elif model_name == "inception_v3":
        model = torchvision.models.inception_v3(pretrained=True,transform_input=True)
    elif model_name == "resnext50_32x4d":
        model = torchvision.models.resnext50_32x4d(pretrained=True) 
    elif model_name == "alexnet":
        model = torchvision.models.alexnet(pretrained=True)

    elif model_name == "mobilenet_v3_large":
        model = torchvision.models.mobilenet.mobilenet_v3_large(pretrained=True)
    elif model_name == 'DenseNet121':
        model = torchvision.models.densenet121(pretrained=True)
    elif model_name == "DenseNet161":
        model = torchvision.models.densenet161(pretrained=True)
    elif model_name == 'mobilenet_v2':
        model = torchvision.models.mobilenet_v2(pretrained=True)
    elif model_name == "shufflenet_v2_x1_0":
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    elif model_name == 'GoogLeNet':
        model = torchvision.models.googlenet(pretrained=True)
    # timm models
    elif model_name == "efficientnet_b0":
        model = timm.create_model('efficientnet_b0', pretrained=True)
    elif model_name == "inception_resnet_v2":
        model = timm.create_model("inception_resnet_v2", pretrained=True)
    elif model_name == "inception_v3_timm":
        model = timm.create_model("inception_v3", pretrained=True)
    elif model_name == "inception_v4_timm":
        model = timm.create_model("inception_v4", pretrained=True)
    elif model_name == "xception":
        model = timm.create_model("xception", pretrained=True)
    # https://github.com/microsoft/robust-models-transfer
    elif model_name == "resnet18_l2_eps0_1":
        m, _ = Madry_model_utils.make_and_restore_model(arch='resnet18', dataset=Madry_ImageNet(''),
                                                        resume_path=dir_path + "/microsoft_robust_models_transfer/src/checkpoints/resnet18_l2_eps0.1.ckpt")
        model = m.model
    elif model_name == "resnet18_l2_eps1":
        m, _ = Madry_model_utils.make_and_restore_model(arch='resnet18', dataset=Madry_ImageNet(''),
                                                        resume_path=dir_path + "/microsoft_robust_models_transfer/src/checkpoints/resnet18_l2_eps1.ckpt")
        model = m.model
    elif model_name == "resnet50_l2_eps0_1":
        m, _ = Madry_model_utils.make_and_restore_model(arch='resnet50', dataset=Madry_ImageNet(''),
                                                        resume_path=dir_path + "/microsoft_robust_models_transfer/src/checkpoints/resnet50_l2_eps0.1.ckpt")
        model = m.model
    elif model_name == "resnet50_l2_eps1":
        m, _ = Madry_model_utils.make_and_restore_model(arch='resnet50', dataset=Madry_ImageNet(''),
                                                        resume_path=dir_path + "/microsoft_robust_models_transfer/src/checkpoints/resnet50_l2_eps1.ckpt")
        model = m.model


    elif model_name == "resnet50_l2_eps0_01":
        m, _ = Madry_model_utils.make_and_restore_model(arch='resnet50', dataset=Madry_ImageNet(''),
                                                        resume_path=dir_path + "/microsoft_robust_models_transfer/src/checkpoints/resnet50_l2_eps0.01.ckpt")
        model = m.model

    elif model_name == "resnet50_l2_eps0_03":
        m, _ = Madry_model_utils.make_and_restore_model(arch='resnet50', dataset=Madry_ImageNet(''),
                                                        resume_path=dir_path + "/microsoft_robust_models_transfer/src/checkpoints/resnet50_l2_eps0.03.ckpt")
        model = m.model
    elif model_name == "resnet50_l2_eps0_05":
        m, _ = Madry_model_utils.make_and_restore_model(arch='resnet50', dataset=Madry_ImageNet(''),
                                                        resume_path=dir_path + "/microsoft_robust_models_transfer/src/checkpoints/resnet50_l2_eps0.05.ckpt")
        model = m.model
    elif model_name == "resnet50_l2_eps0_25":
        m, _ = Madry_model_utils.make_and_restore_model(arch='resnet50', dataset=Madry_ImageNet(''),
                                                        resume_path=dir_path + "/microsoft_robust_models_transfer/src/checkpoints/resnet50_l2_eps0.25.ckpt")
        model = m.model
    elif model_name == "resnet50_l2_eps0_5":
        m, _ = Madry_model_utils.make_and_restore_model(arch='resnet50', dataset=Madry_ImageNet(''),
                                                        resume_path=dir_path + "/microsoft_robust_models_transfer/src/checkpoints/resnet50_l2_eps0.5.ckpt")
        model = m.model
    elif model_name == "resnet50_linf_eps0_5":
        m, _ = Madry_model_utils.make_and_restore_model(arch='resnet50', dataset=Madry_ImageNet(''),
                                                        resume_path=dir_path + "/microsoft_robust_models_transfer/src/checkpoints/resnet50_linf_eps0.5.ckpt")
        model = m.model
    elif model_name == "resnet50_linf_eps1_0":
        m, _ = Madry_model_utils.make_and_restore_model(arch='resnet50', dataset=Madry_ImageNet(''),
                                                        resume_path=dir_path + "/microsoft_robust_models_transfer/src/checkpoints/resnet50_linf_eps1.0.ckpt")
        model = m.model
    elif model_name == "resnet50_linf_eps2_0":
        m, _ = Madry_model_utils.make_and_restore_model(arch='resnet50', dataset=Madry_ImageNet(''),
                                                        resume_path=dir_path + "/microsoft_robust_models_transfer/src/checkpoints/resnet50_linf_eps2.0.ckpt")
        model = m.model
    elif model_name == "resnet50_linf_eps4_0":
        m, _ = Madry_model_utils.make_and_restore_model(arch='resnet50', dataset=Madry_ImageNet(''),
                                                        resume_path=dir_path + "/microsoft_robust_models_transfer/src/checkpoints/resnet50_linf_eps4.0.ckpt")
        model = m.model
    elif model_name == "wide_resnet50_2_l2_eps0_1":
        m, _ = Madry_model_utils.make_and_restore_model(arch='wide_resnet50_2', dataset=Madry_ImageNet(''),
                                                        resume_path=dir_path + "/microsoft_robust_models_transfer/src/checkpoints/wide_resnet50_2_l2_eps0.1.ckpt")
        model = m.model
    elif model_name == "wide_resnet50_2_l2_eps1":
        m, _ = Madry_model_utils.make_and_restore_model(arch='wide_resnet50_2', dataset=Madry_ImageNet(''),
                                                        resume_path=dir_path + "/microsoft_robust_models_transfer/src/checkpoints/wide_resnet50_2_l2_eps1.ckpt")
        model = m.model
    elif model_name == "wide_resnet50_4_l2_eps0_1":
        m, _ = Madry_model_utils.make_and_restore_model(arch='wide_resnet50_4', dataset=Madry_ImageNet(''),
                                                        resume_path=dir_path + "/microsoft_robust_models_transfer/src/checkpoints/wide_resnet50_4_l2_eps0.1.ckpt")
        model = m.model
    elif model_name == "wide_resnet50_4_l2_eps1":
        m, _ = Madry_model_utils.make_and_restore_model(arch='wide_resnet50_4', dataset=Madry_ImageNet(''),
                                                        resume_path=dir_path + "/microsoft_robust_models_transfer/src/checkpoints/wide_resnet50_4_l2_eps1.ckpt")
        model = m.model
    # timm Transformer-based models
    elif model_name == "vit_base_patch16_224":
        model = timm.create_model("vit_base_patch16_224", pretrained=True)
    elif model_name == "levit_384":
        model = timm.create_model("levit_384", pretrained=True)
    elif model_name == "convit_base":
        model = timm.create_model("convit_base", pretrained=True)
    elif model_name == "twins_svt_base":
        model = timm.create_model("twins_svt_base", pretrained=True)
    elif model_name == "pit":
        model = timm.create_model('pit_s_224', pretrained=True)
    else:
        raise ValueError(f"Not supported model name. {model_name}")
    return model

def load_model_cifar10(model_name):
    # https://github.com/huyvnphan/PyTorch_CIFAR10
    if model_name == "vgg11_bn":
        model = cifar10_vgg.vgg11_bn(pretrained=True)
    elif model_name == "vgg13_bn":
        model = cifar10_vgg.vgg13_bn(pretrained=True)
    elif model_name == "vgg16_bn":
        model = cifar10_vgg.vgg16_bn(pretrained=True)
    elif model_name == "vgg19_bn":
        model = cifar10_vgg.vgg19_bn(pretrained=True)
    elif model_name == "resnet18":
        model = cifar10_resnet.resnet18(pretrained=True)
    elif model_name == "resnet34":
        model = cifar10_resnet.resnet34(pretrained=True)
    elif model_name == "resnet50":
        model = cifar10_resnet.resnet50(pretrained=True)
    elif model_name == "mobilenet_v2":
        model = cifar10_mobilenetv2.mobilenet_v2(pretrained=True)
    elif model_name == "inception_v3":
        model = cifar10_inception.inception_v3(pretrained=True)
    elif model_name == "densenet121":
        model = cifar10_densenet.densenet121(pretrained=True)
    elif model_name == "densenet161":
        model = cifar10_densenet.densenet161(pretrained=True)
    elif model_name == "densenet169":
        model = cifar10_densenet.densenet169(pretrained=True)
    elif model_name == "googlenet":
        model = cifar10_googlenet.googlenet(pretrained=True)
    # https://github.com/zjysteven/DVERGE
    elif model_name == "ens3_res20_dverge":
        args = DVERGE.eval.eval_bbox.get_args()
        args.model_file = dir_path + "/DVERGE/checkpoints/dverge/seed_0/3_ResNet20_eps_0.07/epoch_200.pth"
        model = DVERGE.utils.get_models(args, train=False, as_ensemble=True, model_file=args.model_file, leaky_relu=False)
    elif model_name == "ens3_res20_gal":
        args = DVERGE.eval.eval_bbox.get_args()
        args.model_file =dir_path + "/DVERGE/checkpoints/gal/seed_0/3_ResNet20/epoch_200.pth"
        model = DVERGE.utils.get_models(args, train=False, as_ensemble=True, model_file=args.model_file, leaky_relu=True)
    elif model_name == "ens3_res20_adp":
        args = DVERGE.eval.eval_bbox.get_args()
        args.model_file = dir_path + "/DVERGE/checkpoints/adp/seed_0/3_ResNet20/epoch_200.pth"
        model = DVERGE.utils.get_models(args, train=False, as_ensemble=True, model_file=args.model_file, leaky_relu=False)
    elif model_name == "ens3_res20_baseline":
        args = DVERGE.eval.eval_bbox.get_args()
        args.model_file = dir_path + "/DVERGE/checkpoints/baseline/seed_0/3_ResNet20/epoch_200.pth"
        model = DVERGE.utils.get_models(args, train=False, as_ensemble=True, model_file=args.model_file, leaky_relu=False)
    else:
        raise ValueError(f"Not supported model name. {model_name}")
    model.eval()
    return model


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')



