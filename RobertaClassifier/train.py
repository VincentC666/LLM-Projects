import os
from net import RobertaClassification
from YelpData import YelpDataset
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer,RobertaConfig,RobertaModel
import torch.optim as optim
from common import constants
import json
from sklearn import metrics
import logging

def main():

    # ---------- Generate Parameters ----------------
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 50
    BATCH_SIZE = 50
    MODEL_PATH = constants.MODEL_PATH
    DATAFILE_PATH = constants.CLEAN_DATA_PATH

    # Get labels name from dataset_info from dataset
    file_path = './data/yelp_review_full/train/dataset_info.json'
    with open(file_path, 'r') as f:
        data_info = json.load(f)

    # Create directory to save model training paramaters
    if not os.path.exists('params'):
        os.makedirs('params')

    # Setup logging config
    logging.basicConfig(filename='Model_training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

    # Get All Label name from the dataset
    LABELS = data_info['features']['label']['names']

    # ---------- Load Data ----------------
    train_dataset = YelpDataset(DATAFILE_PATH,MODEL_PATH,'train')
    test_dataset = YelpDataset(DATAFILE_PATH,MODEL_PATH,'test')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,drop_last=True,collate_fn=train_dataset.load_data)
    test_loader =  DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,drop_last=True,collate_fn=test_dataset.load_data)

    # ---------- Load Model ----------------

    # Load Config from pretrained model
    config = RobertaConfig.from_pretrained(MODEL_PATH)
    num_labels = len(LABELS)
    # Change the max_postion_embedding to 1024, and put the model to DEVICE
    config.max_position_embeddings = 1024

    # Initialize model
    model = RobertaClassification(config, num_labels).to(DEVICE)
    print(model)

    # ---------- Initialize Optimizer ----------------

    # Define Optimizer
    optimizer = torch.optim.AdamW(model.parameters())

    # Define loss function
    loss_func = torch.nn.CrossEntropyLoss()

    # Store best validation accuracy and best f1 score
    best_val_acc = 0.0
    best_f1 = 0.0

    #  ---------- Model Training Process----------------

    logging.info('-'*10 + 'Model Training Start ' + '-'*10)
    for epoch in range(EPOCHS):
        for i, (tokens,labels) in enumerate(train_loader):
            # send data to DEVICE
            tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
            # put data to model and get output
            out = model(tokens)
            # compute loss function according to output and labels
            loss = loss_func(out, labels)
            # Optimize parameters based on loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record loss and accuracy for every 5 batches
            if i%5==0:
                out = out.argmax(dim=1)
                acc = (out==labels).sum().item()/len(labels)
                logging.inf(f"Epoch:{epoch}, i:{i}, loss:{loss.item()}, acc:{acc}")

        # Evaluating model by test data
        model.eval()
        # Freeze model parameters
        with torch.no_grad():
            val_loss = 0.0
            val_acc = 0.0
            for i, (tokens,labels) in enumerate(test_loader):
                tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
                out = model(tokens)
                # Get validation loss and validation accuracy
                val_loss += loss_func(out, labels)
                out = out.argmax(dim=1)
                val_acc = (out==labels).item().sum()/len(labels)

            val_loss /= len(test_loader)
            val_acc /= len(test_loader)
            logging.info(f'Validation: loss:{val_loss}, acc:{val_acc}')
        # Generate classification report
        report = metrics.classification_report(labels, out, labels=list(range(num_labels)),
                                               target_names=LABELS)
        logging.info(report)
        # Get f1 scorce for the model
        f1 = metrics.f1_score(labels, out,labels=list(range(num_labels)), average='macro')
        # Save the best f1 score parameters
        if f1>best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "params/best_f1.pth")

        # Save the best accuracy parameters
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "params/best_acc.pth")

    # Save last Epoch's parameters
    torch.save(model.state_dict(), f"params/last_bert.pth")
    logging.info(epoch, f"Epcoch：{epoch} last epoch parameters save ！")
    logging.info('-'*10 + 'Model Training  End ' + '-'*10)


if __name__ == '__main__':
    main()