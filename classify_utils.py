import time
import torch
import os

def classify_val(model, device, val_dataloader, loss_fn):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_dataloader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            val_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(val_dataloader)
    accuracy = 100. * correct / len(val_dataloader.dataset)
    print('val loss: {:.4f} accuracy: {}/{} ({:.3f}%)'.format(
        val_loss, correct, len(val_dataloader.dataset),
        100. * correct / len(val_dataloader.dataset)))
    return val_loss, accuracy


class ClassifyTrainMachine:
    def __init__(self, model, device, epochs, val_start_epoch, train_dataloader, val_dataloader, loss_fn, optimizer,
                 scheduler_m=None, model_save_folder='./'):
        self.model = model.to(device)
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler_m = scheduler_m
        self.epochs = epochs
        self.val_start_epoch = val_start_epoch
        self.model_save_folder = model_save_folder

    def train(self):
        self.model.train()
        self.make_pth_folder()
        accuracy = 0
        print('Training...')
        for epoch_ in range(self.epochs):
            epoch = epoch_ + 1
            start_time = time.time()
            loss_sum = 0
            for data, target in self.train_dataloader:
                data = data.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss_sum += loss.item()
                loss.backward()
                self.optimizer.step()
                if self.scheduler_m is not None:
                    self.scheduler_m.step(epoch)
            epoch_time = time.time() - start_time
            print('epoch: {} \tloss: {:.6f} \tlr: {:.6f} \tepoch_time: {:.2f} seconds'
                  .format(epoch, loss_sum, self.optimizer.param_groups[0]['lr'], epoch_time))
            if epoch >= self.val_start_epoch:
                _, accuracy_now = self.val_while_train()
                if accuracy_now > accuracy:
                    self.save_model(self.model_save_folder, self.model.model_name + '_best.pth', cover=True)
                    accuracy = accuracy_now

        self.save_model(self.model_save_folder, self.model.model_name + '_last.pth')

    def save_model(self, path, name, cover=False):
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = os.path.join(path, name)
        if not cover:
            if os.path.exists(file_path):
                while True:
                    name = name + '(1)'
                    file_path = os.path.join(path, name)
                    if not os.path.exists(file_path):
                        break
        else:
            if os.path.exists(file_path):
                os.remove(file_path)

        torch.save(self.model.state_dict(), file_path)

    def val_while_train(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.val_dataloader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data)
                val_loss += self.loss_fn(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        val_loss /= len(self.val_dataloader)
        accuracy = 100. * correct / len(self.val_dataloader.dataset)
        print('val loss: {:.4f}, accuracy: {}/{} ({:.3f}%)'.format(
            val_loss, correct, len(self.val_dataloader.dataset),
            100. * correct / len(self.val_dataloader.dataset)))
        self.model.train()
        return val_loss, accuracy

    def make_pth_folder(self):
        if not os.path.exists(self.model_save_folder):
            os.makedirs(self.model_save_folder)
        index = 0
        while True:
            index += 1
            folder_name = 'train_' + str(index)
            path_tmp = os.path.join(self.model_save_folder, folder_name)
            if not os.path.exists(path_tmp):
                os.makedirs(path_tmp)
                self.model_save_folder = path_tmp
                break