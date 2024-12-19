import numpy as np
import random
import matplotlib.pyplot as plt

# Fungsi untuk menghitung jarak antara dua titik
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Fungsi untuk menghitung total jarak rute
def total_distance(route, points):
    distance = 0
    for i in range(len(route)):
        distance += calculate_distance(points[route[i]], points[route[(i + 1) % len(route)]])
    return distance

# Fungsi untuk membuat populasi awal
def create_population(size, num_points):
    return [random.sample(range(num_points), num_points) for _ in range(size)]

# Fungsi untuk memilih individu terbaik
def select_best(population, points):
    distances = [total_distance(route, points) for route in population]
    best_index = np.argmin(distances)
    return population[best_index], distances[best_index]

# Fungsi untuk melakukan crossover
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = parent1[start:end]
    
    current_index = end
    for gene in parent2:
        if gene not in child:
            child[current_index] = gene
            current_index = (current_index + 1) % size
            
    return child

# Fungsi untuk melakukan mutasi
def mutate(route, mutation_rate):
    for i in range(len(route)):
        if random.random() < mutation_rate:
            swap_index = random.randint(0, len(route) - 1)
            route[i], route[swap_index] = route[swap_index], route[i]

# Fungsi utama untuk menjalankan algoritma genetika
def genetic_algorithm(points, population_size=100, generations=500, mutation_rate=0.01):
    num_points = len(points)
    population = create_population(population_size, num_points)
    
    for generation in range(generations):
        new_population = []
        best_route, best_distance = select_best(population, points)
        new_population.append(best_route)
        
        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            child = crossover(parent1, parent2)
            mutate(child, mutation_rate)
            new_population.append(child)
        
        population = new_population
        
    best_route, best_distance = select_best(population, points)
    return best_route, best_distance

# Fungsi untuk menghasilkan dataset simulasi
def generate_random_points(num_points):
    return np.random.rand(num_points, 2) * 100  # Titik acak dalam rentang 0-100

# Fungsi untuk menjalankan simulasi dengan berbagai jumlah lokasi
def run_simulation(num_points):
    points = generate_random_points(num_points)
    best_route, best_distance = genetic_algorithm(points)
    
    # Menampilkan hasil
    print(f"Jumlah lokasi: {num_points}")
    print("Rute terbaik:", best_route)
    print("Jarak total:", best_distance)
    
    # Visualisasi rute
    plt.figure(figsize=(10, 6))
    plt.scatter(points[:, 0], points[:, 1], color='red')
    
    # Menggambar rute
    route_points = points[best_route]
    plt.plot(route_points[:, 0], route_points[:, 1], marker='o', color='blue')
    
    plt.title(f"Rute Pengiriman untuk {num_points} Lokasi")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.show()

# Menjalankan simulasi untuk 10, 50, dan 100 lokasi
if __name__ == "__main__":
    for num in [10, 50, 100]:
        run_simulation(num)
