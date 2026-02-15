def scaling_decision(predicted_value):

    if predicted_value > 800:
        return "SCALE UP"
    elif predicted_value < 300:
        return "SCALE DOWN"
    else:
        return "STABLE"
