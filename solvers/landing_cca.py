





def landing_cca(model1, model2, dataloader, device):


    prepare_data_loader.
    prepare optimizer


    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        batch_x = batch_x.to(device)
        logits = model1(batch_x)
        loss = model.loss(logits, batch_y)
        train_loss =+ loss.item() * batch_x.size(0)
        test_loss = 0.


        as1 = model1.forward(x)
        as2 = model2.forward(x)
        evaluate tr(a1.T @ a2)
        optimizer1.back()
        optimizer2.back()

    return Qs1, Qs2, svcs1, svcs2

