import numpy as np

class FaultInjector:
    """
    Penelope PHI-CTRL: Dynamic fault injection wrapper.
    Overrides the plant's control effectiveness or corrupts sensor observations.
    """
    def __init__(self, nominal_B):
        self.nominal_B = nominal_B.copy()
        
    def get_effective_B(self, elevator_health=1.0, throttle_health=1.0):
        """
        Returns a modified B matrix to simulate actuator degradation.
        health=1.0 is nominal. health=0.0 is total failure.
        """
        B_eff = self.nominal_B.copy()
        
        # Scale the elevator control column (index 0)
        B_eff[:, 0] *= elevator_health
        
        # Scale the throttle control column (index 1)
        B_eff[:, 1] *= throttle_health
        
        return B_eff
        
    def apply_sensor_bias(self, true_state, bias_vector):
        """
        Adds a bias offset to the true state (e.g., pitot tube icing affecting u).
        bias_vector must match the shape of the state vector.
        """
        return true_state + np.array(bias_vector)
