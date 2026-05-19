import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import torch
import logging

logger = logging.getLogger(__name__)


def calculate_max_downwind_loc(arr, ign_loc_ymin=None, temperature=1.0):
    """
    Calculate the furthest downwind spread of a fire front over time.

    This function tracks how far the fire has spread in the "downwind" direction 
    (assumed to be eastward) for each timestep. It uses the burn index to identify
    burnt/burning areas and finds the maximum eastward extent of the fire.
    
    Since at time zero, there are no ignitions, we can set the minimum ign location
    on the y axis to the initial timestep, which will facilitate the calculation
    of average and instantaneous ROS.
    
    Finding first nonzero index is described here:
    https://stackoverflow.com/questions/47269390/how-to-find-first-non-zero-value-in-every-column-of-a-numpy-array#:~:text=To%20find%20the%20first%20zeros,for%20use%20in%20the%20function.

    Parameters:
    -----------
    arr : torch.Tensor of shape [B, T, C, H, W]
        Fuel density array where:
        - B is the batch size
        - T is the number of timesteps
        - C is the number of channels
        - H, W are spatial dimensions (height and width)
        
        Assumes "downwind" is along the W dimension (horizontal).
    
    ign_loc_ymin : int, optional
        (This parameter is not used in the vectorized logic but kept for signature)
    
    Returns:
    --------
    torch.Tensor of shape [B, T]
        Maximum downwind (horizontal) location indices for each timestep, for each batch item.

    Notes:
    ------
    - The function assumes a primarily western wind (fire spreading eastward)
    - The function converts fuel density to burn indices before calculation
    - For visualization, these indices can be plotted against time to show
      the progression of the fire front
    """
    # Get dimensions from input array
    B, T, C, H, W = arr.shape

    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr)

    # Convert fuel density to burn indices (0: unburned, 1: burning, 2: burned)
    # Assumes fueldens_to_burnindex handles (B, T, C, H, W) input
    # burn_idx = fueldens_to_burnindex(arr) # Shape: (B, T, C, H, W)
    burn_idx = fueldens_to_burnindex_differentiable(arr) # Shape: (B, T, C, H, W)
    # print(torch.unique(torch.argmax(burn_idx, dim=-2)), torch.unique(torch.argmax(burn_idx, dim=-1)), torch.min(burn_idx))

    # We only care about the surface fuel channel (assuming C=1, or C > 1 and surface is at index 0)
    surface_burn_idx = burn_idx.squeeze(2)  # Shape: (B, T, H, W)

       # 1. Projection: Max along H (vertical) -> 1D Profile along W.
    # "What is the max burn prob in column w?"
    col_max_probs = torch.max(surface_burn_idx, dim=-2).values # (B, T, W)
    
    # 2. Weighted sum of indices
    x_grid = torch.arange(W, device=surface_burn_idx.device, dtype=torch.float32).view(1, 1, W)

    # 3. Create Logits (Scores)
    # We want High Score if: (X is large) AND (Prob is high)
    
    # Mask Penalty: If prob is low, apply a massive penalty to the score.
    # If prob ~ 1.0 -> penalty = 0.
    # If prob ~ 0.0 -> penalty = -100,000.
    mask_penalty = (col_max_probs - 1.0) * 1e5
    
    # Score = Position + Penalty
    scores = x_grid + mask_penalty

    # 4. Apply Softmax to Scores
    # Now, the weights concentrate on the *rightmost* pixel that *has fire*.
    weights = torch.softmax(scores / temperature, dim=-1)
    
    # 5. Weighted Sum
    soft_max_x = torch.sum(weights * x_grid, dim=-1)
    
    # 6. Global Mask 
    # If the whole frame is empty, return 0 instead of the right edge (297).
    frame_max = torch.max(col_max_probs, dim=-1).values
    is_fire_present = torch.sigmoid(50.0 * (frame_max - 0.5))
    # print(is_fire_present, soft_max_x * is_fire_present)
    
    return (soft_max_x * is_fire_present) / W

    # # Initialize output tensor
    # max_downwind_loc = torch.zeros(num_timesteps)
    

    # # If ignition location not provided, determine it from first frame
    # # if ign_loc_ymin is None:
    # #     # Sum burn indices along all dimensions except width (results in width-wise profile)
    # #     # Add 2 to ensure all values are positive (for nonzero detection)
    # #     # Look only at the surface layer (index 0 of last dimension)
    # #     sums = torch.sum(burn_idx[0, 0, :, :, 0], axis=0) + 2

    # #     sums = torch.sum(burn_idx[0,0,:,:,0], axis=0) + 2   # Looking at just the surface fuels 
    # #     # sums[0], sums[-1] = 0.,0. # This is done since the first and last rows/columns are immediately zeroes. 
    # #     logger.debug('This is sums:', sums, sums.shape)
    # #     # ign_loc_ymin = torch.argmax(sums != 0, axis=0)
    # #     non_zero_indices = torch.nonzero(sums, as_tuple=True)[0]
    # #     ign_loc_ymin = non_zero_indices[-1].item()
    
    # for t in range(num_timesteps):
    #     # Calculate width-wise profile by summing across height dimension
    #     width_profile = torch.sum(burn_idx[t, ..., :, :], axis=-2).squeeze()    # (W,)

    #     # Flip the profile to search from the right side (east)
    #     flipped_profile = torch.flip(width_profile, dims=[0])    # (W,)

    #     # Find the first non-zero value from the right (eastmost fire point)
    #     non_zero_from_right = torch.nonzero(flipped_profile, as_tuple=True)[0]

    #     if len(non_zero_from_right) > 0:
    #         # Convert the "distance from right" to "distance from left"
    #         max_downwind_loc[t] = (W - 1) - non_zero_from_right[0]
    #     else:
    #         # No fire detected in this frame
    #         max_downwind_loc[t] = 0

    return max_downwind_loc


# def calculate_horizontal_ROS(arr, cell_size, timestep=1.0, method='instant', ign_loc_ymin=None):
#     """
#     Calculate Rate of Spread (ROS) of fire front using either instantaneous or average method.
    
#     Measures how quickly the fire front advances, defined as total distance traveled 
#     divided by total elapsed time.
    
#     Parameters:
#     -----------
#     arr : torch.Tensor or np.ndarray
#         tensor/array with shape [T, ..., H, W] containing fuel density values.
#     cell_size : tuple(float, float)
#         Cell size in meters for (x, y) directions
#     timestep : float
#         Time interval between consecutive frames in seconds
#     method : str, optional
#         Method to calculate ROS: 'instant' or 'average' (default: 'instant')
#     ign_loc_ymin : float, optional
#         Y-index of the minimum point of the ignition location
    
#     Returns:
#     --------
#     torch.Tensor
#         Rate of Spread in meters/second for each time step.
#         First value is always zero (no movement at t=0).
#     """
#     dx, dy = cell_size  # Extract cell sizes (in metres)
    
#     # Get the maximum downwind fire location (in horizontal/x-direction) for each time step
#     max_downwind_loc = calculate_max_downwind_loc(arr, ign_loc_ymin=ign_loc_ymin)  # (T,)
    
#     # Initialize first time step (no movement at t=0)
#     ros_t0 = torch.zeros_like(max_downwind_loc[0])  # tensor scalar
    
#     if method.lower() == 'instant':
#         # Calculate instantaneous ROS: distance between consecutive time steps
#         ros_values = torch.diff(max_downwind_loc) * dy / timestep   # (T-1,)
    
#     elif method.lower() == 'average':
#         # Calculate for time steps t > 0:
#         # 1. Distance traveled from start to current position
#         # 2. Divide by total elapsed time (current time step index * timestep)
#         distances = [(x - max_downwind_loc[0]) * dy for x in max_downwind_loc[1:]]  # (T-1,)
#         time_elapsed = torch.arange(1., len(distances) + 1) * timestep    # (T-1,)
#         ros_values = torch.tensor(distances, dtype=torch.float32) / time_elapsed  # (T-1,)
    
#     else:
#         raise ValueError("Method must be either 'instant' or 'average'")
    
#     # Combine zero first value with calculated values
#     ros = torch.cat([ros_t0.unsqueeze(0).unsqueeze(0), ros_values.unsqueeze(1)], dim=0)  # (T,)
    
#     return ros


def fueldens_to_burnindex(arr, initial_fuel=None, max_fuel_density=0.7001953):
    """
    Calculate burn index based on ratio of current to initial fuel density.
    
    Categorizes each cell as:
    - 0: Unburned cells (fuel nearly intact, ratio > 0.99)
    - 1: Burning cells (partially consumed fuel, 0.05 < ratio < 0.99)
    - 2: Burned cells (fuel depleted, ratio < 0.05)
    
    Parameters:
    -----------
    arr : torch.Tensor of shape [B, T, ..., H, W]
        Fuel density array with first dimension as time and last two dimensions
        as spatial height and width. There can be any number of dimensions in between.

    initial_fuel : torch.Tensor of shape [B, ..., H, W], optional
        Initial fuel density (t=0). If None, uses the first timepoint in arr.
        Shape should match arr without the time dimension.
    
    max_fuel_density: default value from actual fuel density values in our data
    
    Returns:
    --------
    torch.Tensor of shape [B, T, ..., H, W]
        Burn index with same shape as input arr.
    """
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr)

    # If initial_fuel not provided, use first timestep of arr
    if initial_fuel is None:
        initial_fuel = torch.full_like(arr[:, 0, ...], max_fuel_density)
    
    # Expand initial_fuel to match arr's time dimension for broadcasting
    # Add a new dimension at position 0 (time)
    initial_fuel_expanded = initial_fuel.unsqueeze(1)

    # Calculate ratio of current to initial fuel density
    # The unsqueezed dimension allows broadcasting across all timesteps
    ratio = arr / (initial_fuel_expanded + 1e-8)  # Add small epsilon to prevent division by zero
    
    # Create burn index tensor based on ratio thresholds
    burn_index = torch.zeros_like(ratio)
    
    # Apply conditions:
    # - Unburned (0): ratio > 0.99 (fuel nearly intact)
    # - Burning (1): 0.05 < ratio < 0.99 (partially consumed)
    # - Burned (2): ratio < 0.05 (fuel depleted)
    burn_index = torch.where(ratio > 0.99, 0, 
                 torch.where(ratio < 0.05, 2, 1))
    
    return burn_index

def fueldens_to_burnindex_differentiable(arr, initial_fuel=None, temperature=0.02):
    """
    Creates a differentiable 'burn probability' map.
    
    Uses a temperature-scaled Sigmoid (which is a 2-class Softmax) to mimic 
    a hard Argmax threshold.
    
    Args:
        temperature: Controls the sharpness of the decision.
                     - High (e.g. 1.0): Smooth, blurry transition (Soft).
                     - Low (e.g. 0.01): Sharp, binary-like transition (Hard/Argmax).
    """
    # Avoid division by zero
    eps = 1e-6
    
    if initial_fuel is None:
        initial_fuel = arr[:, 0:1, ...] 
        
    ratio = arr / (initial_fuel + eps)
    
    # Threshold for "Burned/Burning" vs "Unburned"
    # ratio < 0.95 means it's starting to burn
    threshold = 0.95
    
    # --- The "Softmax to Argmax" Logic ---
    # Sigmoid(x) is mathematically identical to Softmax([x, 0]).
    # We scale the input by (1 / temperature). 
    # As temperature -> 0, this approaches a Heaviside step function (Hard Threshold).
    
    # difference > 0 means "Burned", difference < 0 means "Unburned"
    difference = threshold - ratio
    
    # Apply temperature scaling
    burn_prob = torch.sigmoid(difference / temperature)
    
    return burn_prob

# def calculate_burned_area(arr, cell_size, units='perc', boundary_correction=796):
#     """
#     Calculates the total burned area for a fire simulation at each time step.
    
#     This function analyzes the surface fuels to determine how much area has burned
#     by counting cells that are either completely burned or currently burning.
    
#     Parameters:
#     -----------
#     arr : torch.Tensor
#         Tensor with shape (B, T, C, H, W) containing fuel density values.
#     cell_size : tuple(float, float)
#         Cell dimensions (dx, dy) in meters for the x and y directions.
#     units : str, optional
#         Output units for the burned area. Options:
#         - 'perc': percentage of total domain area (default)
#         - 'm2': square meters
#         - 'acres': acres (1 m² = 0.000247105 acres)
#     boundary_correction : int or None, optional
#         Number of boundary cells to exclude from the calculation (???).
#         If None, the function will not apply any boundary correction.
    
#     Returns:
#     --------
#     torch.Tensor
#         Burned area, shape (B, T), for each item and time step.
    
#     Notes:
#     ------
#     The burn_index values represent:
#         0: Unburned cells - No fire activity
#         1: Burning cells - Currently on fire
#         2: Burned cells - Completely burned out
#     """
#     batch_size = arr.shape[0]
#     num_timesteps = arr.shape[1]
#     H, W = arr.shape[-2:]

#     if isinstance(arr, np.ndarray):
#         arr = torch.from_numpy(arr)
    
#     # Convert fuel density to burn index
#     burn_idx = fueldens_to_burnindex(arr)
    
#     # Count burned and burning cells for each time step
#     burned_area = torch.zeros((batch_size, num_timesteps), dtype=torch.float32)

#     for t in range(num_timesteps):
#         # Count cells that are completely burned (2) or currently burning (1)
#         num_burned_cells = torch.sum((burn_idx[:, t] == 2).float(), axis=[-2,-1])
#         num_burning_cells = torch.sum((burn_idx[:, t] == 1).float(), axis=[-2,-1])
#         burned_area[:, t] = num_burned_cells.squeeze() + num_burning_cells.squeeze() #- boundary_correction

#     # print(torch.max(burned_area), H, W)
#     # Convert to requested units
#     if units == 'perc':
#         burned_area = burned_area / (H * W)
#     elif units == 'm2':
#         dx, dy = cell_size
#         burned_area = burned_area * dx * dy
#     elif units == 'acres':
#         meter_to_acre = 0.000247105
#         burned_area = burned_area * dx * dy * meter_to_acre
#     else:
#         raise ValueError(f"Unsupported unit: {units}. Use 'perc', 'm2', or 'acres'.")

#     return burned_area

def calculate_burned_area(arr, cell_size, units='perc'):
    """
    Calculates burned area using a differentiable weighted sum.
    """
    # arr shape: (B, T, C, H, W) or (B, T, 1, H, W)
    B, T, C, H, W = arr.shape
    dx, dy = cell_size
    
    # 1. Get differentiable burn map
    # burn_probs = fueldens_to_burnindex(arr)
    burn_probs = fueldens_to_burnindex_differentiable(arr)
    
    # 2. Sum probabilities over spatial dimensions (H, W)
    # This allows gradients to flow back to every pixel that contributed to the sum.
    burned_area_pixels = torch.sum(burn_probs, dim=[-2, -1]).squeeze(-1) # Shape (B, T)
    
    # 3. Convert units (Multiplication is differentiable)
    if units == 'perc':
        total_pixels = float(H * W)
        return (burned_area_pixels) / total_pixels
    elif units == 'm2':
        return burned_area_pixels * (dx * dy)
    elif units == 'acres':
        meter_to_acre = 0.000247105
        return burned_area_pixels * (dx * dy) * meter_to_acre
    else:
        raise ValueError(f"Unknown unit: {units}")
    
    return burned_area_pixels

def get_centroid(arr):
    """
    Calculate the centroid of a group of points
    Input: (n,2) tensor or array of coordinates
    Output: tuple of (x,y) coordinates of the centroid
    """
    # Convert to tensor if not already
    # if not isinstance(arr, torch.Tensor):
    arr = torch.tensor(arr, dtype=torch.float32)

    # Calculate centroid (more efficient than separate sum operations)
    centroid = torch.mean(arr, dim=0)
    
    # Return as tuple (x, y)
    return centroid[0].item(), centroid[1].item()


def calculate_alpha_shape(points, alpha, only_outer=True):
    # Function to calculate the burning/burned area's alpha shape/alpha-concave hull
    # Input: (n,2) list of burning/burned coordinates, alpha parameter, and a boolean to determine whether only the outer border is
    #        returned or if inner edges of the shape are also returned
    #        Default only_outer is currently set to True, but the default should be changed to False if handling complex geometries
    #        or merging fire fronts.
    # Output: set of (i,j) indices in the input array that correspond to the alpha-shape boundary
    # Note -- At least four points are required for this algorithm to work.
    # logger.debug(points.shape)

    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        if (i, j) in edges or (j, i) in edges:
            # Already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # If both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges


def extract_boundary_coords(point_array, centroid, alpha=1):
    """
    Extract the boundary coordinates of burned cells using alpha shapes.
    
    This function:
    1. Centers the burned cell coordinates around their centroid
    2. Computes the alpha shape (concave hull) of the points
    3. Extracts the boundary coordinates from the alpha shape edges
    4. Sorts and organizes the boundary points
    
    Parameters:
    -----------
    point_array : ndarray, shape (num_burned_cells, 2)
        Array of (x, y) indices of burned cells
    centroid : tuple or list of shape (2,)
        The (x, y) coordinates of the centroid of the burned cells
    alpha : float, default=1
        The alpha parameter for the alpha shape algorithm - controls the 
        "tightness" of the boundary (smaller values create tighter boundaries)
    
    Returns:
    --------
    boundary_points : ndarray, shape (num_boundary_points, 2)
        Sorted array of (y, x) coordinates representing the boundary of the burned area,
        centered at the centroid and ordered according to sorting criteria
    """
    # if isinstance(point_array, torch.Tensor):
    #     point_array = point_array.numpy()

    # Create a copy to avoid modifying the original array
    centered_points = point_array.clone()

    # Convert centroid to tensor if it's not already
    # if not isinstance(centroid, torch.Tensor):
    #     centroid = torch.tensor(centroid, dtype=point_array.dtype, device=point_array.device)
    
    # Center points around the centroid
    # This shifts the coordinate system to have the centroid at the origin (0,0)
    centered_points[:, 0] = centered_points[:, 0] - centroid[0]  # Center x-coordinates
    centered_points[:, 1] = centered_points[:, 1] - centroid[1]  # Center y-coordinates

    # For alpha shape computation, we need to detach and convert to numpy temporarily
    # as there's no direct PyTorch implementation of alpha shapes
    centered_points_np = centered_points.detach().cpu().numpy()
    
    # Compute the alpha shape (concave hull) of the points
    # only_outer=True extracts only the exterior boundary
    boundary_edges = calculate_alpha_shape(centered_points_np, alpha=alpha, only_outer=True)
    # logger.debug("edges:", boundary_edges)

    # Extract indices from the boundary edges
    edge_indices = torch.tensor([i for i, j in boundary_edges], device=point_array.device)  # (n,2)
    # unique_indices = torch.unique(edge_indices.view(-1))

    # Get the corresponding points from centered_points
    # Switch to (y, x) order when extracting boundary points
    boundary_points = torch.stack([
        centered_points[edge_indices, 1],  # y-coordinates
        centered_points[edge_indices, 0]   # x-coordinates
    ], dim=1)

    # Extract the boundary coordinates, switching to (y, x) order
    # boundary_points = np.array([[centered_points[i, 1], centered_points[i, 0]] for i, j in boundary_edges]) # Store as [y, x]

    if len(boundary_points) == 0:
        return torch.tensor([], device=point_array.device)
    
    # Sort boundary points: first by y (column 0), then by x (column 1)
    # We'll sort using argsort in PyTorch
    _, y_indices = torch.sort(boundary_points[:, 0], descending=False)
    boundary_points = boundary_points[y_indices]
    
    _, x_indices = torch.sort(boundary_points[:, 1], descending=False, stable=True)
    sorted_boundary = boundary_points[x_indices]
    
    # Reverse the order of points
    boundary_points = torch.flip(sorted_boundary, dims=[0])
    
    # Sort boundary points: first by y (column 0), then by x (column 1)
    # Use lexsort for a more efficient combined sort
    # sort_indices = np.lexsort((boundary_points[:, 1], boundary_points[:, 0]))
    # sorted_boundary = boundary_points[sort_indices]
    
    # Sort the boundary points:
    # First by x-coordinates (which are in column 0 of temp)
    # temp = temp[temp[:, 0].argsort()]
    # Then by y-coordinates using a stable sort (mergesort)
    # temp = temp[temp[:, 1].argsort(kind='mergesort')]
    
    # Reverse the order of points
    # boundary_points = np.flip(sorted_boundary, axis=0)
    
    return boundary_points


def match_points(xy1, xy2):
    """
    Match points between two sets of boundary coordinates based on minimum distance.
    
    This function finds the nearest point in xy1 for each point in xy2 using Euclidean distance.
    It then sorts the matches by distance in descending order and removes duplicates.
    
    Parameters:
    -----------
    xy1 : array-like, shape (n1, 2)
        First set of boundary coordinates, where each row contains [x, y]
    xy2 : array-like, shape (n2, 2)
        Second set of boundary coordinates, where each row contains [x, y]
    
    Returns:
    --------
    matching : ndarray, shape (m, 6)
        Array of matched point pairs with columns:
        [x1, y1, x2, y2, distance, angle]
        where:
        - (x1, y1): Coordinates from xy1
        - (x2, y2): Coordinates from xy2
        - distance: Euclidean distance between matched points
        - angle: Angle of the second point relative to y-axis (North) in degrees
                (Currently set to 0 for all points - seems to be unused)
    """
    matching = []

    if isinstance(xy1, torch.Tensor):
        xy1 = xy1.detach().cpu().numpy()
        xy2 = xy2.detach().cpu().numpy()
    
    # Calculate pairwise distances between all points in xy1 and xy2
    # Each element C[i,j] represents the distance between xy1[i] and xy2[j]
    distances = cdist(xy1, xy2)    # (n1, n2)

    if not isinstance(distances, torch.Tensor):
        distances = torch.tensor(distances, dtype=torch.float32)
    
    # For each point in xy2 (column in distances)
    for col in range(distances.shape[1]):
        # Find the index of the closest point in xy1
        row = torch.argmin(distances[:, col])
        
        # Extract coordinates of the matched point pairs
        px1, py1 = xy1[row, 0], xy1[row, 1]  # Coordinates from xy1
        px2, py2 = xy2[col, 0], xy2[col, 1]  # Coordinates from xy2
        
        # Calculate distance between the matched points
        distance = distances[row, col]
        
        # Angle calculation is currently disabled (set to 0)
        # The commented line would calculate the angle relative to North (y-axis)
        angRoS_N = 0
        # angRoS_N = np.arctan2(xy2[c,1], xy2[c,0]) * 180/np.pi + 90
        
        # Store the matched pair information
        matching.append([px1, py1, px2, py2, distance, angRoS_N])
    
    # Remove duplicate matches
    matching = torch.unique(torch.tensor(matching, dtype=torch.float32), dim=0)
    
    # Sort matches by distance in descending order (furthest matches first)
    # matching = matching[matching[:, 4].argsort()[::-1]]
    # Sort matches by distance in descending order (furthest matches first)
    _, sorted_indices = torch.sort(matching[:, 4], descending=True)
    matching = matching[sorted_indices]
    
    return matching

def calculate_fire_ros(arr, cell_size=(1.0, 1.0), temporal_resolution=1.0, **kwargs):
    """
    Calculates Cumulative Average ROS based on the fire's leading edge.
    Computes speed relative to the FIRST frame in the sequence.
    
    Logic:
    ROS[t] = (Loc[t] - Loc[0]) / (t * temporal_resolution)
    """
    B, T, C, H, W = arr.shape
    dx, dy = cell_size
    
    # print(torch.max(arr), torch.min(arr))
    # 1. Get differentiable burn probabilities
    burn_probs = fueldens_to_burnindex_differentiable(arr)
    
    # 2. Calculate Leading Edge (X-coordinate index)
    # Shape: (B, T)
    max_loc_x = calculate_max_downwind_loc(arr)
    
    # 3. Convert to Meters
    max_loc_x_m = max_loc_x * dx
    
    # 4. Calculate Cumulative Average Velocity
    # We compare every timestep t > 0 back to t=0
    
    # The starting location (e.g., t=6 in your example, index 0 here)
    # Shape: (B, 1)
    start_loc = max_loc_x_m[:, 0:1]
    
    # The future locations (e.g., t=7, 8, 9..., indices 1, 2, 3...)
    # Shape: (B, T-1)
    future_locs = max_loc_x_m[:, 1:]
    
    # Total distance traveled from start
    dist = future_locs - start_loc
    # Elapsed time for each step (1, 2, 3...)
    # Shape: (1, T-1)
    steps = torch.arange(1, T, device=arr.device, dtype=torch.float32).unsqueeze(0)
    elapsed_time = steps * temporal_resolution
    
    # Rate = Total Distance / Total Time
    # Shape: (B, T-1)
    ros = dist / elapsed_time
    
    # Pad the first timestep with 0 (or repeat the first calculated ROS) to maintain shape
    # Since ROS at t=0 relative to t=0 is undefined (0/0), we set it to 0.
    ros_padded = torch.cat([torch.zeros(B, 1, device=arr.device), ros], dim=1)
    
    # Create dummy timesteps
    timesteps = torch.arange(T, device=arr.device)
    
    return timesteps, ros_padded
    
# def calculate_fire_ros(
#     arr, 
#     cell_size=(1.0, 1.0),
#     temporal_resolution=1.0,
#     ros_method='horizontal_instant',
#     interval=1,
#     sliding_window=False,
#     direction_method='edge',
#     min_points=4,
#     ign_loc_ymin=None
# ):
#     """
#     Calculate the Rate of Spread (ROS) of fire front at different timestamps.

#     This unified function handles both horizontal (1D) and boundary-based (2D) ROS calculations
#     with customizable time intervals and measurement methods.
    
#     This function:
#     1. Extracts burn indices from fuel density at specified timestamps
#     2. Finds boundary points of burned areas at both timestamps
#     3. Matches corresponding boundary points between timestamps
#     4. Calculates the maximum distance traveled by the fire front
#     5. Computes the rate of spread as distance/time
    
#     Parameters:
#     -----------
#     arr : torch.Tensor or np.ndarray
#         Array with shape [B, T, C, H, W] containing fuel density values over time
#     cell_size : tuple(float, float), default=(1.0, 1.0)
#         Cell size in meters for (x, y) directions
#     temporal_resolution : float, default=1.0
#         Time interval between consecutive frames in seconds
#     ros_method : str, default='horizontal_instant'
#         Type of ROS calculation:
#         - 'horizontal_instant': Calculates ROS based on maximum horizontal fire spread
#         - 'horizontal_average': Calculates ROS based on maximum horizontal fire spread
#         - 'perimeter_displacement': Calculates ROS based on fire boundary displacement
#     interval : int, default=1
#         Number of timesteps between ROS measurements
#     sliding_window : bool, default=False
#         If True, calculate ROS at every timestep using the interval window
#         If False, calculate ROS at fixed intervals
#     direction_method : str, default='edge'
#         For boundary ROS: method to determine direction of spread
#         - 'edge': Uses maximum boundary point displacement
#         - 'centroid': Uses displacement between centroids
#     min_points : int, default=4
#         Minimum number of burned points required for boundary ROS calculation
#     ign_loc_ymin : float, optional
#         Y-index of the minimum point of the ignition location
    
#     Returns:
#     --------
#     timesteps : list or array
#         Timesteps at which ROS was calculated
#     ros_values : torch.Tensor or np.ndarray
#         Rate of Spread values in m/s. Shape: (B, num_intervals)
#     """
#     dx, dy = cell_size

#     # Ensure interval is at least 1
#     interval = max(1, interval)

#     # Get dimensions (B=Batch, T=Time)
#     B, T, C, H, W = arr.shape

#     # Determine which timesteps to calculate ROS for
#     if sliding_window:
#         # e.g., if T=5, interval=1 -> time_indices_start = [0, 1, 2, 3]
#         time_indices_start = torch.arange(0, T - interval, device=arr.device)
#     else:
#         # e.g., if T=5, interval=1 -> time_indices_start = [0, 1, 2, 3]
#         # e.g., if T=5, interval=2 -> time_indices_start = [0, 2]
#         time_indices_start = torch.arange(0, T - interval, interval, device=arr.device)
    
#     time_indices_end = time_indices_start + interval
    
#     timesteps = time_indices_start # These are the t_idx steps    

#     if len(timesteps) == 0:
#         return torch.tensor([], device=arr.device), torch.tensor([], device=arr.device)

#     # Horizontal ROS calculation
#     if ros_method.lower().startswith('horizontal'):
        
#         # Get max downwind location for ALL batch items and ALL time steps
#         # max_downwind_loc shape: (B, T)
#         max_downwind_loc = calculate_max_downwind_loc(arr, ign_loc_ymin=ign_loc_ymin)

#         # Use advanced tensor indexing to get values at the start and end of intervals
#         # Shape of both: (B, num_intervals)
#         loc_start = max_downwind_loc[:, time_indices_start]
#         loc_end = max_downwind_loc[:, time_indices_end]

#         if ros_method.lower() == 'horizontal_instant':
#             # Instantaneous ROS over each interval
#             distance = (loc_end - loc_start) * dy / W  # in metres
#             time_elapsed = interval * temporal_resolution  # in seconds
#             ros_values = distance / time_elapsed    # in m/s
        
#         elif ros_method.lower() == 'horizontal_average':
#             # Average ROS from start (t=0) to the end of each interval
#             # loc_start_all shape: (B, 1) -> broadcasts to (B, num_intervals)
#             loc_start_all = max_downwind_loc[:, 0].unsqueeze(1) 
#             distance = (loc_end - loc_start_all) * dy / W
            
#             # time_elapsed must have shape (num_intervals,)
#             time_elapsed = time_indices_end.float() * temporal_resolution
#             # Ensure time_elapsed is broadcastable to (B, num_intervals)
#             ros_values = distance / time_elapsed.unsqueeze(0)
        
#         else:
#              raise ValueError(f"Unknown horizontal ros_method: {ros_method}")

#     # Boundary-based ROS (This is complex to vectorize and is not fully implemented)
#     elif ros_method.lower() == 'perimeter_displacement':
#         logging.warning("Vectorized 'perimeter_displacement' ROS is not implemented. Returning zeros.")
#         ros_values = torch.zeros(B, len(timesteps), device=arr.device)
        
#     else:
#         raise ValueError("ros_method must be 'horizontal...' or 'perimeter_displacement'")
    
#     # Return timesteps (1D) and ROS values (2D: Batch, num_intervals)
#     return timesteps, ros_values