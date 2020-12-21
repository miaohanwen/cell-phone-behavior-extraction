import argparse
import os
import pickle
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, SequentialSampler
import videotransforms
from dataset import IVBSSDataset, collate_fn
from model import TAL_Net
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cabin_video_dir', type=str, help='cabin video directory')
    parser.add_argument('--face_video_dir', type=str, help='face video directory')
    parser.add_argument('--test_data_path', type=str, help='path to the test data')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--ckp_path', type=str, help='path to the loaded checkpoint')
    parser.add_argument('--num_classes', type=int, help='number of classes')
    parser.add_argument('--weight', type=float, help='the weight of chunk inclusion loss')
#     parser.add_argument('--save_path', type=float, help='path to save figures')
    args = parser.parse_args()
    return args


def load_ckp(ckp_path, model):
    checkpoint = torch.load(ckp_path)
    model.load_state_dict(checkpoint['model'])
    return model


def calculate_confusion_matrix():
    args = get_parse()
    cabin_video_dir = args.cabin_video_dir
    face_video_dir = args.face_video_dir
    test_data_path = args.test_data_path
    batch_size = args.batch_size
    num_classes = args.num_classes
    weight = args.weight
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
    model = load_ckp(args.ckp_path, model)
    model.cuda()
    model.eval()

    print('Start to calculate confusion matrix')
    all_predicts = []
    all_labels = []
    for i, (cabin_imgs, face_imgs, labels, start_labels, end_labels) in enumerate(test_dataloader):
        cabin_imgs = cabin_imgs.cuda()
        face_imgs = face_imgs.cuda()
        with torch.no_grad():
            class_scores, start_scores, end_scores = model(cabin_imgs, face_imgs)
            class_preds = torch.argmax(class_scores, dim=1)
            class_preds = class_preds.cpu().numpy()
            labels = labels.numpy()
            all_predicts.append(class_preds)
            all_labels.append(labels)
    all_predicts = np.concatenate(all_predicts)
    all_labels = np.concatenate(all_labels)
    cf_matrix = confusion_matrix(all_labels, all_predicts)
    normalized_confusion_matrix = confusion_matrix(all_labels, all_predicts, normalize='true')
    return cf_matrix, normalized_confusion_matrix

#     confusion_matrix = np.zeros([num_classes, num_classes])
#     for i, (cabin_imgs, face_imgs, labels, start_labels, end_labels) in enumerate(test_dataloader):
#         cabin_imgs = cabin_imgs.cuda()
#         face_imgs = face_imgs.cuda()
#         labels = labels.cuda()
#         start_labels = start_labels.cuda()
#         end_labels = end_labels.cuda()
#         with torch.no_grad():
#             class_scores, start_scores, end_scores = model(cabin_imgs, face_imgs, labels, start_labels, end_labels, weight)[3:]
#         class_preds = torch.argmax(class_scores, dim=1)
#         num = class_preds.shape[0]
#         for j in range(num):
#             class_label = labels[j]
#             class_pred = class_preds[j]
#             confusion_matrix[(class_label, class_pred)] += 1
#     return confusion_matrix


if __name__ == '__main__':
    cf_matrix, normalized_cf_matrix = calculate_confusion_matrix()
    print(cf_matrix, normalized_cf_matrix)
    sns.set()
    fig, ax = plt.subplots(1)
    p = sns.heatmap(normalized_cf_matrix, annot=cf_matrix, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel('Gound Truth Class')
    plt.ylabel('Predicted Class')
    plt.show()
    p.get_figure().savefig('figures/heatmap1')


