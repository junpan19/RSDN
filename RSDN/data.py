#from load_duf import DataloadFromFolder
from load_test import DataloadFromFolderTest
from torchvision.transforms import Compose, ToTensor

def transform():
    return Compose([
             ToTensor(),
            ])

def get_test_set(data_dir, upscale_factor, test_name):
    return DataloadFromFolderTest(data_dir, upscale_factor, test_name,transform=transform())

