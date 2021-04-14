def task_generator(data = 'Omniglot', ways, shots)
	for current_way in ways:
		character = data.random_select("Language", "Character")
		for current_shot in shots:
			taskx.add(character.random_select("sample"))
			tasky = current_way_no
	return shuffle_together(taskx, tasky)

class neural_network()
	def initialize_parameters()
		feature_size = 784 #28x28
		hidden_size = 200 #given in paper
		lstm = LSTM(input_size = feature_size + ways, hidden_size)
		classifier = layers(hidden_size + memory_per_slot, 512, 256, 64, ways) #200 + 40 => 240
        key1 = Linear(hidden_size, memory_per_slot)
        
   	def forward(X):
   		memory = allocate(128, 40)
   		intialize_weights_to_random_values(read_weights, write_weights, used_weights) # dimension (128, 1)
   		intialize_weights_to_ones(least_used_weights) # dimension (128, 1)
   		
   		for data_point in X:
   		
   			hidden_state, cell_state = lstm(data_point, hidden_state, cell_state) #dimension (200, 1)
   			key = key1(cell_state) #dimension (1, 40)
   			
   			write_weight = sigmoid(alpha)*read_weight + (1-sigmoid(alpha))*least_used_weight #dimension(128, 1)
   			
   			memory += matmul(write_weight, key) #dimension (128, 40)
   			
   			K = cosine_distance(memory, key.transpose()) #dimension (128, 40) * dimension(40, 1) => dimension(128, 1)
   			read_weight = sigmoid(K) #dimension(128, 1)
   			
   			read = matmul(read_weight.transpose(), memory) #dimension(1, 128) * dimension(128, 40) => dimension (1, 40)
   			
   			use_weights = 0.99*use_weights + read_weights + write_weights
   			least_used_weights = use_weights
   			least_used_weights[smallest_use_weight] = 0
   			
   			to_classifier = concatenate(hidden_state.shrink(), read.shrink()) #200 + 40 => 240
   			Y = classifier(to_classifier) #ways
   			Y_list.append(Y)
   			
   		return Y_list
   		
   		def run_one_episode():
   			taskx, tasky = task_generator(ways, shots)
   			hot_enc_tasky = hot_encode(task_y) # dimension(ways)
   			X = combine_curr_taskx_prev_tasky(taskx, tasky_hot_enc) # dimension(784) + dimension(ways)
   			Y = forward(X)
   			backprop(Y, task_y, "SGD", Learn=True)
   			
   		def accuracy(Y, task_y):
   			return find_accuracy_wrt_repeated_occurence(Y, task_y)
   			
 neural_network.run()
