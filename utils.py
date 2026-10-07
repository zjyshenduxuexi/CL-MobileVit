import os
import sys
import json
import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib.pyplot as plt

# ======================================================
#   START: SUPERVISED CONTRASTIVE LOSS IMPLEMENTATION
# ======================================================
class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
# ======================================================
#    END: SUPERVISED CONTRASTIVE LOSS IMPLEMENTATION
# ======================================================


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  
    train_images_label = [] 
    val_images_path = []  
    val_images_label = []  
    every_class_num = []  
    supported = [".jpg", ".JPG", ".png", ".PNG"] 
  
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        images.sort()
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        plt.xticks(range(len(flower_class)), flower_class)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  
            plt.yticks([])  
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch, sup_con_weight):
    model.train()
    # Define loss functions
    loss_function_ce = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    loss_function_supcon = SupConLoss(temperature=0.1)

    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        # data contains ((view1, view2), labels)
        views, labels = data
        view1, view2 = views
        
        images = torch.cat([view1, view2], dim=0) # [2*B, C, H, W]
        labels = labels.to(device)
        sample_num += view1.shape[0]

        # Pass images through the model with contrastive learning enabled
        logits, contrastive_features = model(images.to(device), contrastive_learning=True)

        # --- Calculate Supervised Contrastive Loss ---
        # Normalize features for SupCon loss calculation
        contrastive_features = F.normalize(contrastive_features, dim=1)
        # Split features back into two views
        f1, f2 = torch.split(contrastive_features, [view1.size(0), view2.size(0)], dim=0)
        # Stack them for SupCon loss input: [B, 2, a_dim]
        features_for_supcon = torch.stack([f1, f2], dim=1)
        loss_supcon = loss_function_supcon(features_for_supcon, labels)

        # --- Calculate Cross-Entropy Loss ---
        # Only use the logits from the first view for classification loss
        logits_ce = logits[:view1.size(0)]
        loss_ce = loss_function_ce(logits_ce, labels)
        
        pred_classes = torch.max(logits_ce, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()
        
        # --- Total Loss ---
        loss = loss_ce + sup_con_weight * loss_supcon
        
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = ("[train epoch {}] loss: {:.3f}, acc: {:.3f}, loss_ce: {:.3f}, loss_sc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            loss_ce.item(),
            loss_supcon.item()
        ))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    
    try:
        num_classes = model.classifier.fc.out_features
    except:
        num_classes = len(json.load(open('class_indices.json'))) if os.path.exists('class_indices.json') else 2

    confusion_matrix = torch.zeros(num_classes, num_classes).to(device)
    
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        
        pred = model(images.to(device), contrastive_learning=False)
        pred_classes = torch.max(pred, dim=1)[1]
        
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        
        for t, p in zip(labels.view(-1), pred_classes.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

    avg_loss = accu_loss.item() / (step + 1)
    accuracy = accu_num.item() / sample_num
    
    tp_per_class = torch.diag(confusion_matrix)
    fp_per_class = confusion_matrix.sum(dim=0) - tp_per_class
    fn_per_class = confusion_matrix.sum(dim=1) - tp_per_class
    
    precision_per_class = tp_per_class / (tp_per_class + fp_per_class + 1e-10)
    recall_per_class = tp_per_class / (tp_per_class + fn_per_class + 1e-10)
    f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + 1e-10)
    
    precision = precision_per_class.mean().item()  
    recall = recall_per_class.mean().item()       
    f1 = f1_per_class.mean().item()               

    print(f"\nValidation Metrics - Epoch {epoch}:")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")  
    print(f"Recall: {recall:.4f}")       
    print(f"F1 Score: {f1:.4f}")         
    
    return avg_loss, accuracy