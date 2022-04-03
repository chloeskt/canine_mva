import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def evaluate_loss(net, device, criterion, dataloader):
    net.eval()

    mean_loss = 0
    count = 0

    with torch.no_grad():
        for it, (token_ids,attn_masks,labels) in enumerate(tqdm(dataloader)):
            labels = labels.to(device)
            token_ids = token_ids.to(device)
            attn_masks = attn_masks.to(device)
    
            # Obtaining the logits from the model
            logits = net(input_ids=token_ids, attn_masks=attn_masks)
            mean_loss += criterion(logits.squeeze(0), labels.squeeze(0)).item()
            count += 1

    return mean_loss / count
    
def train_canine_NER(net, device, criterion, optimizer, lr, lr_scheduler, train_loader, val_loader, epochs):

    best_loss = np.Inf
    best_ep = 1
    nb_iterations = len(train_loader)
    log_interval = nb_iterations // 10  # print the training loss 5 times per epoch
    iters = []
    train_losses = []
    val_losses = []

    for ep in range(epochs):

        net.train()
        running_loss = 0.0
        
        for it, (token_ids,attn_masks,labels) in enumerate(tqdm(train_loader)):
            # Clear gradients
            optimizer.zero_grad()

            # Converting to cuda tensors
            labels = labels.to(device)
            token_ids = token_ids.to(device)
            attn_masks = attn_masks.to(device)
    
            # Obtaining the logits from the model
            logits = net(input_ids=token_ids, attn_masks=attn_masks)
            #output_lab = torch.argmax(logits,dim=2)
            # Computing loss

            loss = criterion(logits.squeeze(0), labels.squeeze(0))
            # Backpropagating the gradients
            loss.backward()
            # Optimization step
            # Adjust the learning rate based on the number of iterations.
            
            optimizer.step()


            running_loss += loss.item()

            if (it + 1) % log_interval == 0:  # Print training loss information
                print()
                print(f"Iteration {it+1}/{nb_iterations} of epoch {ep+1} complete. Loss : {loss.data.item()}")

        lr_scheduler.step()
        val_loss = evaluate_loss(net, device, criterion, val_loader)  # Compute validation loss
        print()
        print("Epoch {} complete! Validation Loss : {}".format(ep+1, val_loss))

        train_losses.append(running_loss/nb_iterations)
        val_losses.append(val_loss)

        if val_loss < best_loss:
            print("Best validation loss improved from {} to {}".format(best_loss, val_loss))
            print()
            net_copy = copy.deepcopy(net)  # save a copy of the model
            best_loss = val_loss
            best_ep = ep + 1

    # Saving the model
    path_to_model=f'models/CANINE_lr_{lr}_val_loss_{round(best_loss, 5)}_ep_{best_ep}.pt'
    torch.save(net_copy.state_dict(), path_to_model)
    print("The model has been saved in {}".format(path_to_model))

    del loss
    torch.cuda.empty_cache()
    return net