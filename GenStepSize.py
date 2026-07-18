import numpy as np

epsilon = 1 / 320
T_trunc = 1 / 8
T = 3

def sigma_squared(t):
    return 1 - np.exp(-2 * t)

def sigma(t):
    val = sigma_squared(t)
    return np.sqrt(max(0, val))

def k(t):
    return np.exp(t)

def generate_timestep_scheme(delta_head: float, delta_tail: float) -> np.ndarray:
    s_backward = [T_trunc]
    current_s = T_trunc
    # Pre-calculate the constant denominator for the backward step
    denom_head = sigma(T_trunc) * sigma(T_trunc + epsilon)
    while current_s > 0:
        numerator_head = sigma(current_s) * sigma(current_s + epsilon)
        step = delta_head * (numerator_head / denom_head)
        new_s = current_s - step
        s_backward.insert(0, new_s)
        current_s = new_s
    s_backward[0] = 0
    s_forward = [T_trunc]
    current_s = T_trunc
    denom_tail = (sigma_squared(T_trunc)**2) * (k(T_trunc)**2)
    while current_s < T:
        numerator_tail = delta_tail * ((sigma_squared(current_s)**2) * (k(current_s)**2))
        step = numerator_tail / denom_tail
        new_s = current_s + min(step,0.2)
        s_forward.append(new_s)
        current_s = new_s
    s_forward[-1] = T
    final_timesteps = s_backward[:-1] + s_forward
    return np.array(final_timesteps), len(s_backward)-1


if __name__ == "__main__":
    delta_head=1/80
    delta_tail=1/320
    # Generate the time-step scheme
    time_steps,cutoff_length = generate_timestep_scheme(delta_head,delta_tail)
    # Print the final result
    print("\n" + "="*40)
    print("Generated Time-step Scheme (as a NumPy array):")
    print(time_steps)
    print(f"\nTotal number of time-steps: {len(time_steps)}")
    print("="*40)
    print(f"\nCutoff_length: {cutoff_length}")
    np.save('timestep.npy',time_steps)

