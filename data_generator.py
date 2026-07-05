import pandas as pd
import numpy as np
import os

def generate_earthquake_data(n_samples=5000, seed=42):
    """
    Generates a realistic synthetic earthquake dataset with geological,
    spatial, historical, and environmental features.
    
    Includes some highly predictive features, some moderately predictive features,
    and several 'junk' or noise features (to test feature selection).
    """
    np.random.seed(seed)
    
    # 1. Geographic coordinates (focused around a highly active seismic zone like Japan/Pacific Ring of Fire)
    # Latitude: 30.0 to 45.0
    latitude = np.random.uniform(30.0, 45.0, n_samples)
    # Longitude: 130.0 to 145.0
    longitude = np.random.uniform(130.0, 145.0, n_samples)
    
    # 2. Geological & Seismic Features (Predictive)
    # Depth of hypocenter (km): 5.0 to 300.0 km
    depth = np.random.exponential(scale=50.0, size=n_samples) + 5.0
    depth = np.clip(depth, 5.0, 300.0)
    
    # Distance to nearest major fault line (km): 0.1 to 150.0 km
    distance_to_fault = np.random.exponential(scale=30.0, size=n_samples) + 0.1
    distance_to_fault = np.clip(distance_to_fault, 0.1, 150.0)
    
    # Fault slip rate (mm/year): 0.5 to 50.0 mm/yr
    fault_slip_rate = np.random.lognormal(mean=2.0, sigma=0.8, size=n_samples)
    fault_slip_rate = np.clip(fault_slip_rate, 0.5, 50.0)
    
    # Rock rigidity / Shear Modulus (GPa): 10.0 to 60.0 GPa
    rock_rigidity = np.random.uniform(10.0, 60.0, size=n_samples)
    
    # Historical earthquake count (previous 10 years, mag > 3.0 within 50km radius)
    historical_eq_count = np.random.poisson(lam=25.0, size=n_samples)
    # Scale based on proximity to fault
    historical_eq_count = historical_eq_count * (100.0 / (distance_to_fault + 5.0))
    historical_eq_count = np.round(np.clip(historical_eq_count, 0, 150)).astype(int)
    
    # Tectonic plate boundary indicator (binary: 1 = boundary zone, 0 = intraplate)
    # More likely to be 1 if close to fault
    boundary_prob = np.exp(-distance_to_fault / 20.0)
    tectonic_plate_boundary = np.random.binomial(n=1, p=boundary_prob)
    
    # P-S wave travel time delay (seconds)
    # Physics relationship: delay is roughly proportional to distance from hypocenter (depth & distance_to_fault)
    hypocentral_distance = np.sqrt(depth**2 + distance_to_fault**2)
    # average speed difference leads to ~ 0.12 seconds delay per km of distance
    p_s_wave_delay = hypocentral_distance * 0.12 + np.random.normal(0, 1.0, n_samples)
    p_s_wave_delay = np.clip(p_s_wave_delay, 0.5, None)
    
    # Seismic station density (sensors within 50km radius)
    seismic_station_density = np.random.randint(1, 25, size=n_samples)
    
    # 3. Noise / Junk Features (Non-predictive / environmental variables to test Evolutionary Feature Selection)
    # Soil moisture content (%) - irrelevant for earthquake magnitude
    soil_moisture = np.random.uniform(5.0, 95.0, n_samples)
    
    # Air temperature (Celsius) - irrelevant
    air_temperature = np.random.uniform(-10.0, 40.0, n_samples)
    
    # Ambient sound levels (dB) - irrelevant
    ambient_noise = np.random.normal(45.0, 8.0, n_samples)
    
    # Tidal height (meters) - irrelevant
    tidal_height = np.random.uniform(-2.0, 2.0, n_samples)
    
    # Random noise feature - complete white noise
    random_noise_feature = np.random.normal(0.0, 1.0, n_samples)
    
    # 4. Target Variable: Earthquake Magnitude
    # We define a physics-inspired non-linear relation:
    # Larger magnitude is associated with higher slip rate, higher rigidity (more stress buildup),
    # shallower depth (for felt magnitude, but let's make actual magnitude relate to slip & rigidity & boundary),
    # and higher historical activity.
    base_magnitude = (
        2.5 
        + 1.1 * np.log10(fault_slip_rate) 
        + 0.03 * rock_rigidity 
        + 0.5 * tectonic_plate_boundary
        + 0.4 * np.log10(historical_eq_count + 1)
        - 0.05 * np.log10(distance_to_fault + 1)
        - 0.001 * depth
    )
    
    # Add random variations / noise to magnitude (standard standard deviation of 0.25)
    magnitude_noise = np.random.normal(loc=0.0, scale=0.25, size=n_samples)
    magnitude = base_magnitude + magnitude_noise
    
    # Keep magnitude values within a realistic range of 2.0 to 8.5
    magnitude = np.clip(magnitude, 2.0, 8.5)
    magnitude = np.round(magnitude, 2)
    
    # Construct DataFrame
    df = pd.DataFrame({
        'latitude': latitude,
        'longitude': longitude,
        'depth': depth,
        'distance_to_fault': distance_to_fault,
        'fault_slip_rate': fault_slip_rate,
        'rock_rigidity': rock_rigidity,
        'historical_eq_count': historical_eq_count,
        'tectonic_plate_boundary': tectonic_plate_boundary,
        'p_s_wave_delay': p_s_wave_delay,
        'seismic_station_density': seismic_station_density,
        'soil_moisture': soil_moisture,
        'air_temperature': air_temperature,
        'ambient_noise': ambient_noise,
        'tidal_height': tidal_height,
        'random_noise_feature': random_noise_feature,
        'magnitude': magnitude
    })
    
    return df

if __name__ == '__main__':
    print("Generating synthetic earthquake dataset...")
    df = generate_earthquake_data(n_samples=5000)
    
    os.makedirs('C:/Users/manoj/.gemini/antigravity/scratch/earthquake_evolutionary_prediction', exist_ok=True)
    output_path = 'C:/Users/manoj/.gemini/antigravity/scratch/earthquake_evolutionary_prediction/earthquake_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Dataset generated and saved successfully to {output_path}")
    print(f"Shape: {df.shape}")
    print("\nFeature Summary:")
    print(df.describe().T[['mean', 'min', 'max']])
