import jsbsim
import numpy as np

class JSBSimPlant6DOF:
    """
    Penelope PHI-CTRL: High-Fidelity Layer 3 6-DOF Plant Bridge.
    Interfaces directly with the NASA-verified JSBSim flight dynamics engine.
    """
    def __init__(self, model_name="c172x", dt=0.02):
        self.dt = dt
        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm.set_dt(self.dt)
        
        success = self.fdm.load_model(model_name)
        if not success:
            raise RuntimeError(f"Failed to load JSBSim model: {model_name}")
            
        self.fdm.disable_output()

    def reset(self, altitude_ft=1000.0, u_fps=120.0):
        """Resets the simulation with standard initial conditions using JSBSim properties."""
        self.fdm['ic/h-sl-ft'] = altitude_ft
        self.fdm['ic/vc-kts'] = u_fps * 0.592484  # Convert fps to knots
        self.fdm['ic/gamma-deg'] = 0.0
        
        self.fdm.run_ic()
        return self.get_state()

    def step(self, control_inputs):
        """
        Advances the 6-DOF simulation by one timestep.
        control_inputs: [aileron, elevator, rudder, throttle] (normalized [-1, 1])
        """
        aileron, elevator, rudder, throttle = control_inputs
        
        self.fdm["fcs/elevator-cmd-norm"] = float(elevator)
        self.fdm["fcs/aileron-cmd-norm"] = float(aileron)
        self.fdm["fcs/rudder-cmd-norm"] = float(rudder)
        self.fdm["fcs/throttle-cmd-norm"] = float(throttle)
        
        self.fdm.run()
        
        return self.get_state()

    def get_state(self):
        """
        Extracts the 12-state vector telemetry for the 6-DOF system:
        [u, v, w, p, q, r, phi, theta, psi, h]
        """
        u = self.fdm["velocities/u-fps"]
        v = self.fdm["velocities/v-fps"]
        w = self.fdm["velocities/w-fps"]
        
        p = self.fdm["velocities/p-rad_sec"]
        q = self.fdm["velocities/q-rad_sec"]
        r = self.fdm["velocities/r-rad_sec"]
        
        phi = self.fdm["attitude/roll-rad"]
        theta = self.fdm["attitude/pitch-rad"]
        psi = self.fdm["attitude/psi-rad"]  # <-- Corrected property name
        
        h = self.fdm["position/h-sl-ft"]
        
        return np.array([u, v, w, p, q, r, phi, theta, psi, h], dtype=np.float32)