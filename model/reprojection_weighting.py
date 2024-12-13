class AdaptiveLossWeighting:
    def __init__(self, epsilon_v=1e-3, epsilon_a=1e-4, max_lambda=1, step_size=0.05):
        assert epsilon_v > 0, "epsilon_v must be greater than zero."
        assert epsilon_a > 0, "epsilon_a must be greater than zero."
        
        self.prev_loss = None
        self.prev_velocity = 0  # Initialized to 0
        self.epsilon_v = epsilon_v
        self.epsilon_a = epsilon_a
        self.lambda_val = 0
        self.max_lambda = max_lambda
        self.step_size = step_size

    def update_lambda(self, current_loss):
        if self.prev_loss is None:
            self.prev_loss = current_loss
            return self.lambda_val

        # Calculate velocity and acceleration
        velocity = current_loss - self.prev_loss
        acceleration = velocity - self.prev_velocity

        # Update lambda based on velocity and acceleration thresholds
        if abs(velocity) < self.epsilon_v and abs(acceleration) < self.epsilon_a:
            if self.lambda_val + self.step_size <= self.max_lambda:
                self.lambda_val += self.step_size

        # Update tracked values
        self.prev_loss = current_loss
        self.prev_velocity = velocity

        return self.lambda_val
    

    """UNDER DEVELOPMENT"""
