import torchvision.transforms as transforms

class CLIPTransform(object):
    def __init__(self, mode='train'):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if mode == 'train':
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])

    def __call__(self, image):
        return self.transforms(image)