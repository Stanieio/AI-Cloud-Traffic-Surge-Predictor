from scaling.auto_scaler import decide_scaling




predicted_value = 8500  # change this to test

servers, message = decide_scaling(predicted_value)

print("Predicted Traffic:", predicted_value)
print("Servers Required:", servers)
print("Scaling Decision:", message)
