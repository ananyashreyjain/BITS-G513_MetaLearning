def task_generator(data = 'Euclidean', ways, shots, classes)
	for current_way_no in range(0, ways):
		way = data.random_select("class", classes)
		for current_shot in shots:
			taskx.add(way.random_select("sample"))
			tasky = current_way_no
	return shuffle_together(taskx, tasky)

class neural_network()
	def initialize_parameters()
		feature_size = 20
		hidden_size = 24
		lstm = LSTM(input_size = feature_size + ways, hidden_size)
		classifier = layers(hidden_size + memory_per_slot, ways)
        keyr = Linear(2*hidden_size, memory_per_slot)
        keyw = Linear(2*hidden_size, memory_per_slot)
        alpha = Linear(2*hidden_size, memory_slots)
        
   	def forward(X):
   		memory = allocate(20, 24)
   		intialize_weights_to_random_values(read_weights, write_weights, used_weights) # dimension (20, 1)
   		intialize_weights_to_ones(least_used_weights) # dimension (20, 1)
   		
   		for data_point in X:
   		   	least_used_weights = use_weights.copy()
   			least_used_weights[smallest_use_weight] = 0
   			
   			hidden_state, cell_state = lstm(data_point, hidden_state, cell_state) # dimension (24, 1)
   			key = concatenate([hidden_state, cell_state]).transpose() # dimension (1, 48)
   			
   			write_weight = sigmoid(alpha(key))*read_weight + (1-sigmoid(alpha(key)))*least_used_weight # dimension(20, 1)
   			
   			memory += matmul(write_weight, keyw(key)) # dimension (20, 24)
   			
   			K = cosine_distance(memory, keyr(key).transpose()) # dimension (20, 24) * dimension(24, 1) => dimension(20, 1)
   			read_weight = sigmoid(K) # dimension(20, 1)
   			
   			read = matmul(read_weight.transpose(), memory) # dimension(1, 20) * dimension(20, 24) => dimension (1, 24)
   			
   			use_weights = 0.95*use_weights + read_weights + write_weights
   			
   			to_classifier = concatenate(hidden_state.shrink(), read.shrink()) # 24 + 24 => 48
   			Y = classifier(to_classifier) # ways
   			Y_list.append(Y)
   			
   		return Y_list
   		
   		def run_one_episode():
   			taskx, tasky = task_generator(ways, shots)
   			hot_enc_tasky = hot_encode(task_y) # dimension(ways)
   			X = combine_curr_taskx_prev_tasky(taskx, tasky_hot_enc) # dimension(20) + dimension(ways)
   			Y = forward(X)
   			backprop(Y, task_y, "RMSprop", Learn=True)
   			
   		def accuracy(Y, task_y):
   			return find_accuracy_wrt_repeated_occurence(Y, task_y)
   			
neural_network.run()


