def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    total_loss = 0
    idx = 0
    for data in dataloader:
        idx+=1
        
        info(f"----Batch {idx}----")
        input_tensor, target_tensor = data['image'].to(device), data['formula'].to(device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        encoder_output = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_output, target_tensor)
        
        # print(encoder_output.device, 'My device')
        
        # print(f"Decoder OutDim : {decoder_outputs.shape}, Target Tensor Dim: {target_tensor.shape}")
        # print(f"Target tensor: {target_tensor[0][0]}")
        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
        
        print(f'Loss for batch {idx} = {loss.item()}')

    return total_loss / len(dataloader)

def train(train_dataloader, encoder, decoder, n_epochs, save_interval = 2, learning_rate=0.001, print_every=1, plot_every=5):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss().to(device) #as stated in assignment
    
    # Print model's device
    # print("Encoder's device:", next(encoder.parameters()).device)

    pb = tqdm(range(1, n_epochs + 1))
    for epoch in pb:
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            
        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        
        if epoch % save_interval == 0:
            saveModel(f'checkpoints/encoder_epoch_{epoch}.pt', encoder.state_dict(), encoder_optimizer.state_dict(), loss)
            saveModel(f'checkpoints/decoder_epoch_{epoch}.pt', decoder.state_dict(), decoder_optimizer.state_dict(), loss)
            

        pb.set_description('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg))
        
    showPlot(plot_losses)