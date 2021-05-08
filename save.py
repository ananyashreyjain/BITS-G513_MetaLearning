class neural_network(torch.nn.Module):
    def __init__(self, feature_size=5*4, hidden_size=26, ways = 0, shots=0, mem=(20, 26), nreads=2, batch=1):
        super(neural_network, self).__init__()
        self.l1 = torch.nn.LSTM(input_size = feature_size + ways, hidden_size = hidden_size).to(device)
        self.d0 = torch.nn.Linear(feature_size + ways, hidden_size).to(device)
        self.d1 = torch.nn.Linear(hidden_size + nreads*mem[-1], ways).to(device)
        self.kr = torch.nn.Linear(2*hidden_size, mem[-1] * nreads).to(device)
        self.kw = torch.nn.Linear(2*hidden_size, mem[-1] * nreads).to(device)
        self.alpha = torch.nn.Linear(2*hidden_size, mem[0]).to(device)
        self.feature_size = feature_size
        self.ways = ways
        self.shots = shots
        self.hidden_size = hidden_size
        self.stored_dataset = None
        self.mem = mem
        self.nreads = nreads
        self.batch = batch
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=5e-5, momentum=0.9)
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #self.optimizer, patience=2000, factor=0.5, min_lr=5e-6)
        self.criterion = torch.nn.NLLLoss()

    def backprop(self, Y, labels):
        loss = self.criterion(Y, labels)
        loss.backward()
        self.optimizer.step()
        #self.scheduler.step(loss)
        self.optimizer.zero_grad()
        return loss

    def forward(self, X, g, MANN, LSTM, wu, wr, lstmh,  lstmc, memory):
        if LSTM:
            op, (lstmh,  lstmc) = self.l1(X, (lstmh,  lstmc)) # shape=> [1, batch, hidden_size]
        else:
            lstmh = self.d0(X[-1].unsqueeze(0))
        if MANN:
            key = torch.cat((lstmh.reshape(self.batch, -1),  lstmc.reshape(self.batch, -1)), dim=1)# shape => [batch, 2*hidden_size]
            #least recently used
            wlu = wu.clone().detach() # shape => [batch, slots, nreads]
            k_sm_wu = wu.sort(dim=1)[1] # shape => [batch, slots, nreads]
            rem_indx = (k_sm_wu < self.nreads)
            wlu[rem_indx] = 1
            wlu[~rem_indx] = 0

            #write
            write_key = torch.tanh(self.kw(key)).reshape(self.batch, self.nreads, -1) #shape => [batch, nreads, mem_size]
            sigma = torch.tanh(self.alpha(key)) #shape => [batch, slots]
            ww = (sigma*(wr.permute(2,0,1)) + (1-sigma)*(wlu.permute(2,0,1))).permute(1,2,0) # shape => [batch, slots, nreads]
            memory = memory +ww@write_key  #shape => [batch, slots, mem_size]
            #read
            read_key = torch.tanh(self.kr(key)).reshape(self.batch, self.nreads, -1)  #shape => [batch, nreads, mem_size]
            read_key_norm = torch.sqrt(torch.sum(torch.square(read_key), dim=-1)) #shape => [batch, nreads]
            memory_norm = torch.sqrt(torch.sum(torch.square(memory), dim=[1,-1])) # shape => [batch]
            dist_prod = ((read_key_norm.permute(1,0))* memory_norm).permute(1,0) # shape => [batch, nreads]

            z = memory@(read_key.permute((0,2,1))) #shape => [batch, slots, nreads]
            z = (z.permute(1,0,2)/dist_prod).permute(1,0,2) #shape => [batch, slots, nreads]
            wr = torch.nn.functional.softmax(z, dim=1) #shape => [batch, slots, nreads]
            rt = wr.permute((0,2,1))@memory #shape => [batch, nreads, mem_size]

            wu = g*wu + wr + ww # shape => [batch, slots, nreads]
        else:
            rt = torch.rand((self.batch, self.nreads, memory.size()[-1]))
        y_lstm_1_last = torch.cat((lstmh.reshape(self.batch, -1), rt.reshape(self.batch, -1)), dim=1) # shape => [batch,  hidden_size + nreads*mem_size]
        y_d1 = torch.nn.functional.relu(self.d1(y_lstm_1_last)) # shape => [batch, ways]
        pred = torch.nn.functional.log_softmax(y_d1, dim=-1) #shape => [batch, ways]
        return pred, wu, wr, lstmh,  lstmc, memory

    def batched_training(self, X, y_test, g, MANN, LSTM, train):
        memory = 1e-6*torch.ones(self.batch, *self.mem).to(device) # shape => [batch, slots, mem_size]
        wr = torch.randint(0, 2,(self.batch, self.mem[0], self.nreads)).to(device) # shape => [batch, slots, nreads]
        wu = torch.randint(0, 2,(self.batch, self.mem[0], self.nreads)) .to(device)# shape => [batch, slots, nreads]
        lstmh,  lstmc = torch.zeros((1,self.batch,self.hidden_size)).to(device), torch.zeros((1,self.batch,self.hidden_size)).to(device) # shape => [1, batch, hidden_state]
        train_loss = 0
        y_batch = torch.empty((self.batch, 0, self.ways)).to(device)
        for index in range(0, X.size()[0], 1):
            x = X[index: index + 1] #shape => [1, batch, features]
            y, wu, ww, lstmh, lstmc, memory = self.forward(x, g, MANN, LSTM, wu, wr, lstmh,  lstmc, memory)
            y_batch = torch.cat((y_batch, y.unsqueeze(1)), dim=1)  #shape => [shots*ways, batch, ways]
        if train:
            train_loss = self.backprop(y_batch.reshape(-1, self.ways), y_test. reshape(-1))
        return train_loss, y_batch

    def run_episodes (self, data, task_creator, MANN, LSTM, update_dataset = True, train=True):
        if self.stored_dataset == None or update_dataset:
            self.stored_dataset = task_creator(data, self.ways, self.shots, self.batch)
        data_train_x, data_train_y = self.stored_dataset
        data_train_x = torch.from_numpy(data_train_x).to(device) # shape => [batch, (shotsxways), featuresX, featuresY]
        data_train_y = torch.from_numpy(data_train_y).to(device) # shape => [batch, (shotsxways)]
        y_test = data_train_y # shape => [batch, (shotsxways)]
        batch_X = torch.empty((self.ways*self.shots, 0, self.feature_size + self.ways)).to(device)
        for cur_batch in range(self.batch):
            y_prev = torch.tensor([0]).to(device) #shape = [batch]
            X = torch.empty((0, self.feature_size + self.ways)).to(device)
            for index, curr_x in enumerate(data_train_x[cur_batch]):
                y_prev_henc = torch.eye(self.ways).to(device) # shape => [ways, ways]
                y_prev_henc = y_prev_henc[y_prev].squeeze() # shape => [ways]
                flattened_x = curr_x.reshape(-1) # shape => [features]
                curr_X = torch.cat((flattened_x, y_prev_henc)).reshape(1, -1) # shape => [1, features + ways]
                X = torch.cat((X, curr_X)) # shape => [(shotsxways), features]
                y_prev = data_train_y[0][index] # shape => scalar
            X = torch.unsqueeze(X, 1) #shape => [(shotsxways), 1, features]
            X = X.float()
            batch_X = torch.cat((batch_X, X), dim=1) #shape => [(shotsxways), batch, features]
        episode_loss, pred= self.batched_training(batch_X, y_test, 0.95, MANN, LSTM, train)
        return X, y_test, torch.exp(pred), episode_loss

    def accuracy(self, y_pred, labels):
        prev_labels = {}
        acc_dict = {}
        y_pred = y_pred.detach().numpy()
        labels = labels.detach().numpy()
        for index, label in enumerate(labels):
            pred_dec = np.argmax(y_pred[index]).item()
            label_dec = labels[index]
            if prev_labels.get(label_dec, None)  != None:
                prev_labels[label_dec] += 1
                if pred_dec == label_dec:
                    acc_dict[label_dec].append(1)
                else:
                    acc_dict[label_dec].append(0)
            else:
                prev_labels[label_dec] = 1
                if pred_dec == label_dec:
                    acc_dict[label_dec] = [1]
                else:
                    acc_dict[label_dec] = [0]
        return acc_dict

nn = neural_network(ways = 2, shots = 5)
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    
print(count_parameters(nn))
print(nn)
