def test(data_matrix, data_labels, test_indices, nn):
    total_accuracy = 0
    for j in range(100):
        correct_guess_count = 0
        for i in test_indices:
            test_sample = data_matrix[i]
            prediction = nn.predict(test_sample)
            if data_labels[i] == prediction:
                correct_guess_count += 1

        total_accuracy += (correct_guess_count / len(test_indices))
    return total_accuracy / 100

# Try various number of hidden nodes and see what performs best
for hidden_nodes in range(5, 50, 5):
    nn = OCRNeuralNetwork(hidden_nodes, data_matrix, data_labels, train_indices, False)
    accuracy = test(data_matrix, data_labels, test_indices, nn)
    print(f"{hidden_nodes} Hidden Nodes: {accuracy:.4f}")
