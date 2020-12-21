import argparse
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, SequentialSampler
import videotransforms
from dataset import IVBSSDataset, collate_fn
from model import TAL_Net


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cabin_video_dir', type=str, help='cabin video directory')
    parser.add_argument('--test_data_path', type=str, help='path to the test data')
#     parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--ckp_path', type=str, help='path to the loaded checkpoint')
    parser.add_argument('--num_classes', type=int, help='number of classes')
    args = parser.parse_args()
    return args


# def load_ckp(ckp_path, model):
#     checkpoint = torch.load(ckp_path)
#     model.load_state_dict(checkpoint['model'])
#     return model


def predict():
    args = get_parse()
    cabin_video_dir = args.cabin_video_dir
    test_data_path = args.test_data_path
#     batch_size = args.batch_size
    num_classes = args.num_classes
        
    print('Start to load data')
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224),
                                         videotransforms.ToTensor()
                                         ])
    test_dataset = IVBSSDataset(face_video_dir,
                                cabin_video_dir,
                                test_data_path,
                                test_transforms
                               )
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 sampler=SequentialSampler(test_dataset),
                                 collate_fn=collate_fn
                                )

    model = TemporalActionLocalization(num_classes, pretrained_I3D_model) 
    print('Load checkpoint')
    model = load_ckp(args.ckp_path, model)

    model.cuda()
    model.eval()

    print('Start to test')
    test_loss = 0.0
    test_steps = 0
    for i, (face_imgs, cabin_imgs, labels) in enumerate(test_dataloader):
        face_imgs = face_imgs.cuda()
        cabin_imgs = cabin_imgs.cuda()
        for k, v in labels.items():
            labels[k] = v.cuda() 
        loss = model(face_imgs, cabin_imgs, labels)
        test_loss += loss.item()
        test_steps += 1
    avg_test_loss = test_loss/test_steps
    return avg_test_loss    


if __name__ == '__main__':
    test()

