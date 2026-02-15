def decide_scaling(predicted_traffic):

    if predicted_traffic < 3000:
        servers = 1
        action = "Low traffic - No scaling needed"

    elif 3000 <= predicted_traffic < 7000:
        servers = 2
        action = "Medium traffic - Scale to 2 servers"

    else:
        servers = 3
        action = "High traffic detected! Scaling to 3 servers"

    return servers, action
