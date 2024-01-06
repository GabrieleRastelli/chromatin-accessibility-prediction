import os.path

import torch
from torch import optim, nn
from tqdm import tqdm

from public.earlyStopping import EarlyStopping
from public.evaluation import Evaluation

from torch.nn.parallel import DataParallel


class Builder:

    def __init__(self, model, model_name):
        """
        :param model:
        :param model_name:
        """
        if model is None or model_name is None:
            raise ValueError('the model or the model_name is None!')

        self.cwd = "performance"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model
        self.model_name = model_name

        pretrained_model_path = self.cwd + "/pth/" + self.model_name + ".pth"
        if os.path.exists(pretrained_model_path):
            print('Loading pretrained model...')
            saved_state_dict = torch.load(pretrained_model_path, map_location=self.device)
            filtered_state_dict = {k: v for k, v in saved_state_dict.items() if
                                   k in model.state_dict() and v.shape == model.state_dict()[k].shape}
            self.model.load_state_dict(filtered_state_dict, strict=False)

        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            self.model = DataParallel(self.model)  # Wrap the model with DataParallel

        self.model = self.model.to(device=self.device)
        self.model_name = model_name
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=0.0001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                              patience=3,
                                                              verbose=True)
        self.loss_function = nn.MSELoss()

        self.batch_size = 32
        self.epochs = 50
        self.learn_loader = None
        self.val_loader = None
        self.infer_loader = None

    def learn(self, learn_loader, val_loader):
        """
        :param fine_tune:
        :param learn_loader:
        :param val_loader:
        :return:
        """
        early_stopping = EarlyStopping(patience=5,
                                       verbose=True)
        for epoch in range(self.epochs):
            self.model.train()
            ProgressBar = tqdm(learn_loader)
            for sample in ProgressBar:
                self.optimizer.zero_grad()

                ProgressBar.set_description("Learning_Epoch %d" % epoch)
                sequence, label = sample
                sequence, label = sequence.to(self.device), label.to(
                    self.device)  # Ensure data is on the correct device
                y = self.model(sequence.float())

                loss = self.loss_function(y, label.float())
                ProgressBar.set_postfix(loss=loss.item())

                loss.backward()
                self.optimizer.step()

            val_loss = []

            self.model.eval()
            with torch.no_grad():
                for val_sequence, val_label in val_loader:
                    val_sequence, val_label = val_sequence.to(self.device), val_label.to(
                        self.device)  # Ensure data is on the correct device
                    val_y = self.model(val_sequence.float())
                    val_label = val_label.float()

                    val_loss.append(self.loss_function(val_y, val_label).item())

                val_loss_avg = torch.mean(torch.Tensor(val_loss))

            self.scheduler.step(val_loss_avg)

            early_stopping(val_loss=val_loss_avg,
                           model=self.model,
                           path=self.cwd + "/pth/" + self.model_name + ".pth")

    def inference(self, infer_loader, measure=True):
        """
        :param selected:
        :param infer_loader:
        :param measure:
        :return:
        """
        y_label = []
        ground_label = []

        self.model.eval()
        ProgressBar = tqdm(infer_loader)
        for sample in ProgressBar:
            ProgressBar.set_description("Inferring")

            sequence, label = sample
            sequence, label = sequence.to(self.device), label.to(self.device)
            y = self.model(sequence.float())

            y_label.append(y.squeeze(dim=0).detach().cpu().numpy())
            ground_label.append(label.squeeze(dim=0).detach().cpu().numpy())

        if measure:
            if not os.path.exists(self.cwd + "/index"):
                os.makedirs(self.cwd + "/index")
            if not os.path.exists(self.cwd + "/index_batch"):
                os.makedirs(self.cwd + "/index_batch")
            evaluation = Evaluation(save_dir=self.cwd + "/index/" + self.model_name + "_performance.index",
                                    save_alldata=self.cwd + '/index_batch/' + self.model_name + '_performance.index')

            evaluation.measure_accessibility_prediction(y_label=y_label,
                                                        ground_label=ground_label)

    def __call__(self, mode):
        if mode not in ["learn", "infer", "all"]:
            raise ValueError('the mode must in ["learn", "infer", "all"]')

        if mode == "all" or mode == "learn":
            self.learn(learn_loader=self.learn_loader,
                       val_loader=self.val_loader)

        if mode == "all" or mode == "infer":
            self.inference(infer_loader=self.infer_loader,
                           measure=True)
