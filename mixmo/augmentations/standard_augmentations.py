from torchvision import transforms
from mixmo.utils.logger import get_logger

LOGGER = get_logger(__name__, level="DEBUG")


class CustomCompose(transforms.Compose):

    def __call__(self, img, apply_postprocessing=True):
        for i, t in enumerate(self.transforms):

            if i == len(self.transforms) - 1 and not apply_postprocessing:
                return {"pixels": img, "postprocessing": t}

            img = t(img)
        return img


cifar_mean = (0.4913725490196078, 0.4823529411764706, 0.4466666666666667)
cifar_std = (0.2023, 0.1994, 0.2010)


def get_default_composed_augmentations(dataset_name):
    if dataset_name.startswith("cifar"):
        normalize = transforms.Normalize(cifar_mean, cifar_std)
        # Transformer for train set: random crops and horizontal flip
        train_transformer = CustomCompose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
                # postprocessing
                CustomCompose([transforms.ToTensor(), normalize])
            ]
        )
        test_transformer = CustomCompose([
            transforms.ToTensor(),
            normalize])

    elif dataset_name.startswith("tinyimagenet"):
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        train_transformer = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, padding=4),
                CustomCompose([transforms.ToTensor(), normalize])
            ]
        )
        test_transformer = CustomCompose([transforms.ToTensor(), normalize])

    else:
        raise ValueError(dataset_name)

    return train_transformer, test_transformer
