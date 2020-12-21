import argparse
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, SequentialSampler
import videotransforms
from dataset import IVBSSDataset, collate_fn
from model import TAL_Net
import time


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cabin_video_dir', type=str, help='cabin video directory')
    parser.add_argument('--face_video_dir', type=str, help='face video directory')
    parser.add_argument('--test_data_path', type=str, help='path to the test data')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--ckp_path', type=str, help='path to the loaded checkpoint')
    parser.add_argument('--num_classes', type=int, help='number of classes')
    parser.add_argument('--weight', type=float, help='the weight of chunk inclusion loss')
    args = parser.parse_args()
    return args


def test():
    args = get_parse()
    cabin_video_dir = args.cabin_video_dir
    face_video_dir = args.face_video_dir
    test_data_path = args.test_data_path
    batch_size = args.batch_size
    num_classes = args.num_classes
    weight = args.weight

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('Start to load data')

    test_transforms = transforms.Compose([videotransforms.CenterCrop(224),
                                          videotransforms.ToTensor(),
                                          videotransforms.ClipNormalize()
                                          ])
    test_dataset = IVBSSDataset(cabin_video_dir,
                                face_video_dir,
                                test_data_path,
                                test_transforms
                                )
    print('Total number of test samples is {0}'.format(len(test_dataset)))
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 sampler=SequentialSampler(test_dataset),
                                 collate_fn=collate_fn
                                 )
    model = TAL_Net(num_classes)

    print('Load checkpoint')
    ckp = torch.load(args.ckp_path)
    model.load_state_dict(ckp['model'])
    model.to(device)
    model.eval()

    print('Start to test')
    test_loss = 0.0
    test_class_loss = 0.0
    test_chunk_inclusion_loss = 0.0
    class_accuracy = 0.0
    test_steps = 0

    #     start_time = time.time()
    for i, (cabin_imgs, face_imgs, labels, start_labels, end_labels) in enumerate(test_dataloader):
        cabin_imgs = cabin_imgs.to(device)
        face_imgs = face_imgs.to(device)
        labels = labels.to(device)
        start_labels = start_labels.to(device)
        end_labels = end_labels.to(device)
        with torch.no_grad():
            loss, class_loss, chunk_inclusion_loss, class_scores, start_scores, end_scores = model(
                cabin_imgs, face_imgs, labels, start_labels, end_labels, weight)
        test_loss += loss.item()
        test_class_loss += class_loss.item()
        test_chunk_inclusion_loss += chunk_inclusion_loss.item()
        class_pred = torch.argmax(class_scores, dim=1)
        class_accuracy += torch.sum((class_pred == labels).float()) / labels.shape[0]
        test_steps += 1

    avg_test_loss = test_loss / test_steps
    avg_test_class_loss = test_class_loss / test_steps
    avg_test_chunk_inclusion_loss = test_chunk_inclusion_loss / test_steps
    avg_class_accuracy = class_accuracy / test_steps

    #     end_time = time.time()
    #     total_time = end_time-start_time
    #     avg_time = total_time/(test_steps*batch_size)

    print('avg_test_loss:{0}, avg_test_class_loss:{1}, avg_test_chunk_inclusion_loss:{2}, avg_class_accuracy:{3}'.format( avg_test_loss, avg_test_class_loss, avg_test_chunk_inclusion_loss, avg_class_accuracy))


if __name__ == '__main__':
    test()