import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import optuna

# 1. Load and prepare the data
def load_data(file_path):
    # Assuming data is in CSV format
    data = pd.read_csv(file_path)
    
    # Define input and output features
    input_features = ['DIE_AREA', 'CORE_AREA', 'FP_CORE_UTIL', 'PL_TARGET_DENSITY']
    output_features = ['wire_length', 'DIEAREA_mm^2', 'power', 'Congestion']
    
    X = data[input_features]
    y = data[output_features]
    
    return X, y, input_features, output_features

# 2. Train a model to predict output parameters from input parameters
def train_prediction_model(X, y):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Train a random forest regressor for multi-output regression
    base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    model = MultiOutputRegressor(base_model)
    
    model.fit(X_train_scaled, y_train_scaled)
    
    # Evaluate the model
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")
    
    return model, scaler_X, scaler_y

# 3. Objective function for optimization
def create_objective_function(model, scaler_X, scaler_y, weights):
    def objective(params):
        # Scale the input parameters
        params_scaled = scaler_X.transform(params.reshape(1, -1))
        
        # Predict the output parameters
        outputs_scaled = model.predict(params_scaled)
        
        # Inverse transform to get the actual output values
        outputs = scaler_y.inverse_transform(outputs_scaled)
        
        # Calculate weighted cost (negative for minimization)
        # Assuming lower values are better for all outputs
        weighted_cost = (
            weights[0] * outputs[0, 0] +  # wirelength
            weights[1] * outputs[0, 1] +  # overall_die_area
            weights[2] * outputs[0, 2] +  # power
            weights[3] * outputs[0, 3]    # congestion
        )
        
        return weighted_cost
    
    return objective

# 4. Optimize using Optuna for more sophisticated search
def optimize_parameters_optuna(model, scaler_X, scaler_y, input_ranges, weights, n_trials=100):
    def objective(trial):
        params = np.array([
            trial.suggest_float('DIE_AREA', input_ranges[0][0], input_ranges[0][1]),
            trial.suggest_float('CORE_AREA', input_ranges[1][0], input_ranges[1][1]),
            trial.suggest_float('FP_CORE_UTIL', input_ranges[2][0], input_ranges[2][1]),
            trial.suggest_float('PL_TARGET_DENSITY', input_ranges[3][0], input_ranges[3][1])
        ])
        
        # Scale the input parameters
        params_scaled = scaler_X.transform(params.reshape(1, -1))
        
        # Predict the output parameters
        outputs_scaled = model.predict(params_scaled)
        
        # Inverse transform to get the actual output values
        outputs = scaler_y.inverse_transform(outputs_scaled)
        
        # Calculate weighted cost
        weighted_cost = (
            weights[0] * outputs[0, 0] +  # wirelength
            weights[1] * outputs[0, 1] +  # overall_die_area
            weights[2] * outputs[0, 2] +  # power
            weights[3] * outputs[0, 3]    # congestion
        )
        
        return weighted_cost
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    best_params = np.array([
        study.best_params['DIE_AREA'],
        study.best_params['CORE_AREA'],
        study.best_params['FP_CORE_UTIL'],
        study.best_params['PL_TARGET_DENSITY']
    ])
    
    # Predict the output for the best parameters
    best_params_scaled = scaler_X.transform(best_params.reshape(1, -1))
    best_outputs_scaled = model.predict(best_params_scaled)
    best_outputs = scaler_y.inverse_transform(best_outputs_scaled)
    
    return best_params, best_outputs, study

# 5. Visualize the results
def visualize_results(study, input_features, output_features):
    # Plot optimization history
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(study.trials)), [t.value for t in study.trials], marker='o', linestyle='-')
    plt.xlabel('Trial')
    plt.ylabel('Objective Value')
    plt.title('Optimization History')
    plt.grid(True)
    plt.savefig('optimization_history.png')
    plt.close()
    
    # Plot parameter importances
    importances = optuna.importance.get_param_importances(study)
    plt.figure(figsize=(10, 6))
    plt.bar(importances.keys(), importances.values())
    plt.xlabel('Parameter')
    plt.ylabel('Importance')
    plt.title('Parameter Importances')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('parameter_importances.png')
    plt.close()
    
    return importances

# 6. Main function
def main():
    # Load data
    file_path = '/home/adi/Documents/Scripts/csv/merged.csv'  # Replace with your dataset path
    X, y, input_features, output_features = load_data(file_path)
    
    # Train prediction model
    model, scaler_X, scaler_y = train_prediction_model(X, y)
    
    # Define parameter ranges for optimization
    # These should be set based on your domain knowledge
    input_ranges = [
        [X['DIE_AREA'].min(), X['DIE_AREA'].max()],
        [X['CORE_AREA'].min(), X['CORE_AREA'].max()],
        [X['FP_CORE_UTIL'].min(), X['FP_CORE_UTIL'].max()],
        [X['PL_TARGET_DENSITY'].min(), X['PL_TARGET_DENSITY'].max()]
    ]
    
    # Define weights for multi-objective optimization
    # These weights should be adjusted based on your priorities
    # e.g., if wirelength is most important, give it the highest weight
    weights = [0.25, 0.25, 0.25, 0.25]  # Equal weights for all outputs
    
    # Run optimization
    best_params, best_outputs, study = optimize_parameters_optuna(
        model, scaler_X, scaler_y, input_ranges, weights, n_trials=200
    )
    
    # Print results
    print("\nOptimal Input Parameters:")
    for i, feature in enumerate(input_features):
        print(f"{feature}: {best_params[i]}")
    
    print("\nPredicted Outputs:")
    for i, feature in enumerate(output_features):
        print(f"{feature}: {best_outputs[0, i]}")
    
    # Visualize results
    importances = visualize_results(study, input_features, output_features)
    print("\nParameter Importances:")
    for param, importance in importances.items():
        print(f"{param}: {importance}")
    
    # Save the model
    import joblib
    joblib.dump({
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'input_features': input_features,
        'output_features': output_features
    }, 'floorplan_optimizer_model.pkl')
    
    print("\nModel saved to 'floorplan_optimizer_model.pkl'")
    
if __name__ == "__main__":
    main()