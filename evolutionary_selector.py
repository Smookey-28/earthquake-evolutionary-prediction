import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

class GeneticAlgorithmFeatureSelector:
    def __init__(self, n_features, population_size=20, generations=15, 
                 crossover_rate=0.8, mutation_rate=0.1, tournament_size=3,
                 random_state=42):
        self.n_features = n_features
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        
        # Initialize population randomly (each individual is a boolean array)
        # Ensure that at least some features are selected (not all False)
        self.population = self.rng.choice([True, False], size=(population_size, n_features))
        for i in range(population_size):
            if not np.any(self.population[i]):
                self.population[i, self.rng.choice(n_features)] = True
                
        self.best_individual = None
        self.best_fitness = -float('inf')
        self.history = []  # To track best and average fitness per generation

    def calculate_fitness(self, individual, X_train, X_val, y_train, y_val):
        """
        Evaluate fitness as the R^2 score of a Random Forest model on the validation set.
        To keep GA runtimes fast, we use a moderately sized Random Forest.
        """
        # If no features are selected, return very low fitness
        if not np.any(individual):
            return -10.0
            
        selected_features = np.where(individual)[0]
        X_tr = X_train[:, selected_features]
        X_va = X_val[:, selected_features]
        
        # Fast RF model configuration for efficient GA evaluations
        model = RandomForestRegressor(
            n_estimators=30,
            max_depth=8,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        try:
            model.fit(X_tr, y_train)
            preds = model.predict(X_va)
            r2 = r2_score(y_val, preds)
            
            # Subtly penalize choosing too many features to encourage parsimony
            # (e.g., penalty of 0.005 per active feature)
            penalty = 0.005 * np.sum(individual)
            fitness = r2 - penalty
            
            return fitness
        except Exception as e:
            return -10.0

    def select_parent(self, fitnesses):
        """Tournament selection."""
        candidates = self.rng.choice(self.population_size, size=self.tournament_size, replace=False)
        best_candidate = candidates[np.argmax(fitnesses[candidates])]
        return self.population[best_candidate].copy()

    def crossover(self, parent1, parent2):
        """Uniform crossover."""
        if self.rng.random() < self.crossover_rate:
            mask = self.rng.choice([True, False], size=self.n_features)
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
            return child1, child2
        return parent1.copy(), parent2.copy()

    def mutate(self, individual):
        """Bit-flip mutation."""
        for i in range(self.n_features):
            if self.rng.random() < self.mutation_rate:
                individual[i] = not individual[i]
        
        # Ensure at least one feature remains selected
        if not np.any(individual):
            individual[self.rng.choice(self.n_features)] = True
        return individual

    def fit(self, X, y):
        """Runs the genetic algorithm optimization loop."""
        # Split into training and validation sets for fitness evaluation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Support both DataFrame and numpy array
        X_train_arr = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        X_val_arr = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
        y_train_arr = y_train.values if isinstance(y_train, pd.Series) else y_train
        y_val_arr = y_val.values if isinstance(y_val, pd.Series) else y_val
        
        fitnesses = np.zeros(self.population_size)
        
        print(f"Starting Genetic Algorithm Feature Selection...")
        print(f"Parameters: Pop Size={self.population_size}, Gens={self.generations}, Features={self.n_features}")
        
        for generation in range(self.generations):
            # Calculate fitness for all individuals
            for i in range(self.population_size):
                fitnesses[i] = self.calculate_fitness(
                    self.population[i], X_train_arr, X_val_arr, y_train_arr, y_val_arr
                )
            
            # Find best in this generation
            best_idx = np.argmax(fitnesses)
            gen_best_fitness = fitnesses[best_idx]
            gen_best_individual = self.population[best_idx].copy()
            gen_avg_fitness = np.mean(fitnesses)
            
            # Save history
            self.history.append({
                'generation': generation + 1,
                'best_fitness': gen_best_fitness,
                'avg_fitness': gen_avg_fitness
            })
            
            print(f"Gen {generation+1:02d}/{self.generations:02d} | "
                  f"Best Fitness: {gen_best_fitness:.4f} | "
                  f"Avg Fitness: {gen_avg_fitness:.4f} | "
                  f"Features Chosen: {np.sum(gen_best_individual)}/{self.n_features}")
            
            # Track global best
            if gen_best_fitness > self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_individual = gen_best_individual.copy()
            
            # Create next generation
            next_population = []
            
            # Elitism: Keep the best 2 individuals unchanged
            sorted_indices = np.argsort(fitnesses)[::-1]
            next_population.append(self.population[sorted_indices[0]].copy())
            next_population.append(self.population[sorted_indices[1]].copy())
            
            # Generate offspring to fill the rest of the population
            while len(next_population) < self.population_size:
                parent1 = self.select_parent(fitnesses)
                parent2 = self.select_parent(fitnesses)
                
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                next_population.append(child1)
                if len(next_population) < self.population_size:
                    next_population.append(child2)
            
            self.population = np.array(next_population)
            
        print(f"Genetic Algorithm finished. Optimal subset found.")
        print(f"Best fitness: {self.best_fitness:.4f}")
        return self.best_individual, self.history
